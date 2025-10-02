import os
import time
import apikeys

# Environment flag for prompt logging control
ENABLE_SESSION_PROMPT_LOGGING = os.getenv("ENABLE_SESSION_PROMPT_LOGGING", "true").lower() == "true"

# Optional Langfuse import - app will work without it
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    Langfuse = None
    LANGFUSE_AVAILABLE = False

def get_langfuse_client():
    """Initialize and return Langfuse client"""
    if not LANGFUSE_AVAILABLE:
        return None
    return Langfuse(
        public_key=apikeys.LANGFUSE_PUBLIC_KEY,
        secret_key=apikeys.LANGFUSE_SECRET_KEY,
        host=apikeys.LANGFUSE_HOST
    )

def create_conversation_trace(session_id, metadata=None):
    """Create a new trace for tracking conversations"""
    if not LANGFUSE_AVAILABLE:
        return MockTrace()
    
    try:
        langfuse = get_langfuse_client()
        
        # Use the modern Langfuse API - create a trace ID and update it
        if hasattr(langfuse, 'create_trace_id'):
            trace_id = langfuse.create_trace_id()
            
            # Update the trace with session_id and metadata
            if hasattr(langfuse, 'update_current_trace'):
                langfuse.update_current_trace(
                    session_id=session_id,
                    metadata=metadata or {"app": "policy_copilot"}
                )
            
            # Flush to ensure trace data is sent
            if hasattr(langfuse, 'flush'):
                langfuse.flush()
            
            # Return a simple trace object that we can use
            return LangfuseTraceWrapper(langfuse, trace_id)
        else:
            # Fallback: return a mock trace object
            return MockTrace()
    except Exception as e:
        print(f"Warning: Langfuse trace creation failed: {e}")
        return MockTrace()

def track_llm_call(trace, name, model, input_text, output_text, usage=None, metadata=None, execution_time=None):
    """Track individual LLM calls as generations"""
    try:
        # Add execution time to metadata if provided
        if execution_time is not None:
            if metadata is None:
                metadata = {}
            metadata['execution_time'] = execution_time
        
        # Use the trace wrapper to track generation
        if hasattr(trace, 'track_generation'):
            return trace.track_generation(name, model, input_text, output_text, usage, metadata)
        else:
            return None
    except Exception as e:
        print(f"Warning: Langfuse generation tracking failed: {e}")
        return None

def track_error(trace, error, context=None):
    """Track errors with context"""
    try:
        # Use the trace wrapper to track events
        if hasattr(trace, 'track_event'):
            trace.track_event("application_error", "ERROR", {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {}
            })
    except Exception as e:
        print(f"Warning: Langfuse error tracking failed: {e}")

def track_performance(trace, operation_name, execution_time, metadata=None):
    """Track performance metrics"""
    try:
        # Use the trace wrapper to track events
        if hasattr(trace, 'track_event'):
            trace.track_event(f"{operation_name}_performance", "DEFAULT", {
                "execution_time": execution_time,
                "operation": operation_name,
                **(metadata or {})
            })
    except Exception as e:
        print(f"Warning: Langfuse performance tracking failed: {e}")

def flush_langfuse_data():
    """Flush all pending Langfuse data to ensure it's sent"""
    try:
        if LANGFUSE_AVAILABLE:
            langfuse = get_langfuse_client()
            if langfuse and hasattr(langfuse, 'flush'):
                langfuse.flush()
    except Exception as e:
        print(f"Warning: Langfuse flush failed: {e}")

def truncate_prompt(prompt_text, max_length=2000):
    """Truncate prompt text for metadata storage"""
    if not ENABLE_SESSION_PROMPT_LOGGING:
        return None
    if len(prompt_text) <= max_length:
        return prompt_text
    return prompt_text[:max_length] + "... [truncated]"

def track_session_started(trace, session_id, metadata=None):
    """Track session start event"""
    try:
        if hasattr(trace, 'track_event'):
            trace.track_event(
                "session_started", 
                "INFO", 
                {
                    "session_id": session_id,
                    "timestamp": time.time(),
                    **(metadata or {})
                }
            )
    except Exception as e:
        print(f"Warning: Session start tracking failed: {e}")

def track_session_ended(trace, session_id, metadata=None):
    """Track session end event"""
    try:
        if hasattr(trace, 'track_event'):
            trace.track_event(
                "session_ended", 
                "INFO", 
                {
                    "session_id": session_id,
                    "timestamp": time.time(),
                    **(metadata or {})
                }
            )
    except Exception as e:
        print(f"Warning: Session end tracking failed: {e}")

class LangfuseTraceWrapper:
    """Wrapper for Langfuse trace operations"""
    def __init__(self, langfuse_client, trace_id):
        self.langfuse_client = langfuse_client
        self.trace_id = trace_id
    
    def track_generation(self, name, model, input_text, output_text, usage=None, metadata=None):
        """Track a generation within this trace"""
        try:
            if hasattr(self.langfuse_client, 'start_observation'):
                # Use the modern API
                generation = self.langfuse_client.start_observation(
                    name=name,
                    as_type='generation',
                    model=model,
                    input=input_text,
                    output=output_text,
                    usage_details=usage,
                    metadata=metadata or {}
                )
                
                # End the generation to ensure it's properly recorded
                generation.end()
                
                # Flush to ensure data is sent to Langfuse
                if hasattr(self.langfuse_client, 'flush'):
                    self.langfuse_client.flush()
                
                return generation
            elif hasattr(self.langfuse_client, 'start_generation'):
                # Fallback to deprecated API
                generation = self.langfuse_client.start_generation(
                    name=name,
                    model=model,
                    input=input_text,
                    metadata=metadata or {}
                )
                generation.end()
                return generation
        except Exception as e:
            print(f"Warning: Generation tracking failed: {e}")
        return None
    
    def track_event(self, name, level, metadata=None):
        """Track an event within this trace"""
        try:
            if hasattr(self.langfuse_client, 'create_event'):
                self.langfuse_client.create_event(
                    name=name,
                    level=level,
                    metadata=metadata or {}
                )
        except Exception as e:
            print(f"Warning: Event tracking failed: {e}")

class MockTrace:
    """Mock trace object for when Langfuse is not available or fails"""
    def track_generation(self, name, model, input_text, output_text, usage=None, metadata=None):
        return MockGeneration()
    
    def track_event(self, name, level, metadata=None):
        pass
    
    def start_as_current_generation(self, *args, **kwargs):
        return MockGeneration()
    
    def start_generation(self, *args, **kwargs):
        return MockGeneration()
    
    def create_generation(self, *args, **kwargs):
        return MockGeneration()
    
    def generation(self, *args, **kwargs):
        return MockGeneration()
    
    def create_event(self, *args, **kwargs):
        pass
    
    def event(self, *args, **kwargs):
        pass

class MockGeneration:
    """Mock generation object for when Langfuse is not available or fails"""
    def end(self, *args, **kwargs):
        pass
