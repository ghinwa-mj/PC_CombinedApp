import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import json
import apikeys
from langfuse.openai import openai as langfuse_openai
from openai import OpenAI

# Define outcome indicators mapping
OUTCOME_INDICATORS = {
    "Unemployment": {
        "total": "unemployment_rate_total",
        "female": "unemployment_rate_female", 
        "male": "unemployment_rate_male",
        "display_name": "Unemployment Rate",
        "unit": "%"
    },
    "Youth Unemployment": {
        "total": "youth_unemployment_rate_percentage_total",
        "female": "youth_unemployment_rate_percentage_female",
        "male": "youth_unemployment_rate_percentage_male",
        "display_name": "Youth Unemployment Rate",
        "unit": "%"
    },
    "Youth not in Education or Training": {
        "total": "youth_notin_education_employment_training_total",
        "female": "youth_notin_education_employment_training_female",
        "male": "youth_notin_education_employment_training_male", 
        "display_name": "Youth Not in Education, Employment or Training",
        "unit": "%"
    },
    "Stunting": {
        "total": "stunting_prevalence",
        "female": None,  # No gender breakdown available
        "male": None,    # No gender breakdown available
        "display_name": "Stunting Prevalence",
        "unit": "%"
    },
    "Maternal Mortality Ratio": {
        "total": "maternal_mortality_ratio",
        "female": "maternal_mortality_ratio",  # Maternal mortality is female-specific
        "male": None,  # No male data for maternal mortality
        "display_name": "Maternal Mortality Ratio",
        "unit": "per 100,000 live births"
    },
    "Still Birth Rates": {
        "total": "still_birth_rates_per_1000_total_births",
        "female": None,  # No gender breakdown available
        "male": None,    # No gender breakdown available
        "display_name": "Still Birth Rates",
        "unit": "per 1,000 total births"
    },
    "Youth Literacy Rate": {
        "total": "youth_literacy_rate%",
        "female": None,  # No gender breakdown available
        "male": None,    # No gender breakdown available
        "display_name": "Youth Literacy Rate",
        "unit": "% (Aged 15-24)"
    },
    "Neonatal Mortality Rate": {
        "total": "neonatal_mortality_rate_per_1000_live_births",
        "female": None,  # No gender breakdown available
        "male": None,    # No gender breakdown available
        "display_name": "Neonatal Mortality Rate",
        "unit": "per 1,000 live births"
    },
    "HIV Incidence": {
        "total": "hiv_incidence_per_1000_uninfected_population",
        "female": None,  # No gender breakdown available
        "male": None,    # No gender breakdown available
        "display_name": "HIV Incidence",
        "unit": "per 1,000 uninfected population"
    },
    "Victims of Intentional Homicides": {
        "total": "intentional_homicides_per_100000_people",
        "female": None,  # No gender breakdown available
        "male": None,    # No gender breakdown available
        "display_name": "Intentional Homicides",
        "unit": "per 100,000 people"
    },
    "Access to Safe Water Services": {
        "total": "access_to_safe_water_services_percentage",
        "female": None,  # No gender breakdown available
        "male": None,    # No gender breakdown available
        "display_name": "Access to Safe Water Services",
        "unit": "% of Population"
    },
    "Adolescent Pregnancies": {
        "total": "adolescent_fertility_rate_births_per_1000_women_ages_15_19",
        "female": None,  # No gender breakdown available
        "male": None,    # No gender breakdown available
        "display_name": "Adolescent Pregnancies",
        "unit": "births per 1,000 women aged 15-19"
    }}

# Define segmentation options for each indicator
SEGMENTATION_OPTIONS = {
    "Unemployment": {
        "Gender": {
            "total": "unemployment_rate_total",
            "female": "unemployment_rate_female",
            "male": "unemployment_rate_male",
            "labels": ["Total", "Female", "Male"]
        },
        "Age": {
            "total": "unemployment_rate_15_years_old_and_over",
            "youth": "unemployment_rate_15_to_24_years_old",
            "labels": ["Total (15+)", "Youth (15-24)"]
        }
    },
    "Youth not in Education or Training": {
        "Gender": {
            "total": "youth_notin_education_employment_training_total",
            "female": "youth_notin_education_employment_training_female",
            "male": "youth_notin_education_employment_training_male",
            "labels": ["Total", "Female", "Male"]
        }
    },
    "Youth Unemployment": {
        "Gender": {
            "total": "youth_unemployment_rate_percentage_total",
            "female": "youth_unemployment_rate_percentage_female",
            "male": "youth_unemployment_rate_percentage_male",
            "labels": ["Total", "Female", "Male"]
        }
    },
    "Stunting": {
        # No segmentation available
    },
    "Maternal Mortality Ratio": {
        # No segmentation available (female-specific)
    },
    "Still Birth Rates": {
        # No segmentation available
    },
    "Youth Literacy Rate": {
        # No segmentation available
    },
    "Neonatal Mortality Rate": {
        # No segmentation available
    },
    "HIV Incidence": {
        # No segmentation available
    },
    "Intentional Homicides": {
        # No segmentation available
    },
    "Access to Safe Water Services": {
        # No segmentation available
    },
    "Adolescent Pregnancies": {
        # No segmentation available
    }
}

# Page config is handled by main_app.py

# Custom CSS for styling the squares
st.markdown("""
<style>
    .visualization-square {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        border: 2px solid transparent;
    }
    
    .visualization-square:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
        border-color: #ffd700;
    }
    
    .visualization-square h3 {
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .visualization-square p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .main-title {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-title h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-title p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .squares-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .clickable-square {
        cursor: pointer;
    }
    
    .clickable-square:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2) !important;
        border-color: #ffd700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load the cleaned unemployment dataset"""
    try:
        # Try different possible paths for the dataset
        possible_paths = [
            '../Data/cleaned_final_dataset.csv',
            'Data/cleaned_final_dataset.csv',
            '/Users/ghinwamoujaes/Desktop/Policy CoPilot/TechWork/MVP/feature2/Data/cleaned_final_dataset.csv'
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        
        if df is None:
            st.warning("❌ Could not find the cleaned unemployment dataset. Please check the file path.")
            return None
        
        return df
    except Exception as e:
        st.warning(f"❌ Error loading dataset: {str(e)}")
        return None

def load_dataset_summaries():
    """Load dataset summaries from CSV"""
    try:
        # Try different possible paths for the dataset summaries
        possible_paths = [
            '../Data/DatasetSummaries.csv',
            'Data/DatasetSummaries.csv',
            '/Users/ghinwamoujaes/Desktop/Policy CoPilot/TechWork/MVP/feature2/Data/DatasetSummaries.csv'
        ]
        
        summaries_df = None
        for path in possible_paths:
            if os.path.exists(path):
                summaries_df = pd.read_csv(path)
                break
        
        if summaries_df is not None:
            # Create a dictionary mapping developmental outcomes to descriptions
            summaries_dict = dict(zip(summaries_df['Developmental Outcome'], summaries_df['Description of Dataset']))
            return summaries_dict
        else:
            st.warning("⚠️ Dataset summaries file not found")
            return {}
    except Exception as e:
        st.warning(f"⚠️ Could not load dataset summaries: {str(e)}")
        return {}

def get_countries_list(df):
    """Get sorted list of countries from the dataset"""
    if df is not None:
        countries = sorted(df['country'].unique())
        return countries
    return []

def get_countries_with_data(df, indicator_name):
    """Get countries that have data for the selected indicator"""
    if df is None:
        return []
    
    indicator_info = OUTCOME_INDICATORS[indicator_name]
    total_column = indicator_info['total']
    
    # Filter countries that have at least one non-null value for this indicator
    countries_with_data = df[df[total_column].notna()]['country'].unique()
    return sorted(countries_with_data)

def reduce_data_for_ai(data_json, max_records=30, analysis_type="time_trend"):
    """Reduce data size for AI analysis by sampling intelligently"""
    try:
        import json
        
        # Parse the JSON data
        if isinstance(data_json, str):
            data = json.loads(data_json)
        else:
            data = data_json
        
        if not isinstance(data, list):
            return data_json
        
        # If data is small enough, return as is
        if len(data) <= max_records:
            return data_json
        
        # For large datasets, sample intelligently based on analysis type
        if len(data) > max_records:
            # Sort by year if year column exists
            if isinstance(data[0], dict) and 'year' in data[0]:
                data = sorted(data, key=lambda x: x.get('year', 0))
            
            sampled_data = []
            
            if analysis_type == "time_trend":
                # For time trends, focus on key years: start, middle, recent
                if len(data) <= max_records:
                    sampled_data = data
                else:
                    # Take first, middle, and last records with some intermediate samples
                    step = max(1, len(data) // max_records)
                    
                    # Always include first record
                    sampled_data.append(data[0])
                    
                    # Sample middle records
                    for i in range(step, len(data) - step, step):
                        sampled_data.append(data[i])
                    
                    # Always include last record
                    if len(data) > 1:
                        sampled_data.append(data[-1])
                    
                    # Ensure we don't exceed max_records
                    if len(sampled_data) > max_records:
                        sampled_data = sampled_data[:max_records]
            
            elif analysis_type in ["comparison", "peer_comparison"]:
                # For comparisons, ensure we have data for all countries
                if isinstance(data[0], dict) and 'country' in data[0]:
                    # Group by country and sample from each
                    countries = {}
                    for record in data:
                        country = record.get('country', 'Unknown')
                        if country not in countries:
                            countries[country] = []
                        countries[country].append(record)
                    
                    # Sample from each country
                    records_per_country = max(1, max_records // len(countries))
                    for country_data in countries.values():
                        if len(country_data) <= records_per_country:
                            sampled_data.extend(country_data)
                        else:
                            # Sample evenly from this country's data
                            step = len(country_data) // records_per_country
                            for i in range(0, len(country_data), step):
                                sampled_data.append(country_data[i])
                                if len(sampled_data) >= max_records:
                                    break
                        if len(sampled_data) >= max_records:
                            break
                else:
                    # Fallback to general sampling
                    step = len(data) // max_records
                    sampled_data = [data[i] for i in range(0, len(data), step)][:max_records]
            
            else:
                # General sampling for other analysis types
                step = len(data) // max_records
                if step < 1:
                    step = 1
                
                # Always include first and last records
                sampled_data.append(data[0])
                if len(data) > 1:
                    sampled_data.append(data[-1])
                
                # Sample middle records
                for i in range(step, len(data) - step, step):
                    sampled_data.append(data[i])
                    if len(sampled_data) >= max_records:
                        break
                
                # Ensure we don't exceed max_records
                if len(sampled_data) > max_records:
                    sampled_data = sampled_data[:max_records]
            
            return json.dumps(sampled_data)
        
        return data_json
        
    except Exception as e:
        # If reduction fails, return original data
        return data_json

def generate_insights(data_json, indicator_name, country_name=None, countries_list=None, analysis_type="time_trend"):
    """Generate insights using OpenAI GPT-4o based on the visualization data"""
    try:
        # Reduce data size for AI analysis
        reduced_data_json = reduce_data_for_ai(data_json, max_records=30, analysis_type=analysis_type)
        
        # Check if the reduced data is still too large (rough estimate)
        if len(reduced_data_json) > 50000:  # Rough token estimate
            # Try more aggressive reduction
            reduced_data_json = reduce_data_for_ai(data_json, max_records=15, analysis_type=analysis_type)
        
        # Initialize OpenAI client
        client = OpenAI(api_key=apikeys.OPENAI_API_KEY)
        
        # Get indicator information
        indicator_info = OUTCOME_INDICATORS[indicator_name]
        display_name = indicator_info['display_name']
        unit = indicator_info['unit']
        
        # Create context for the prompt
        if analysis_type == "time_trend":
            context = f"Developmental Outcome: {display_name} ({unit})\nCountry: {country_name}\nAnalysis Type: Time trend analysis"
        elif analysis_type == "segmented":
            context = f"Developmental Outcome: {display_name} ({unit})\nCountry: {country_name}\nAnalysis Type: Segmented analysis"
        elif analysis_type == "comparison":
            countries_str = ", ".join(countries_list)
            context = f"Developmental Outcome: {display_name} ({unit})\nCountries: {countries_str}\nAnalysis Type: Country comparison"
        elif analysis_type == "benchmarking":
            context = f"Developmental Outcome: {display_name} ({unit})\nCountry: {country_name}\nAnalysis Type: Global benchmarking analysis"
        elif analysis_type == "peer_comparison":
            context = f"Developmental Outcome: {display_name} ({unit})\nCountry: {country_name}\nAnalysis Type: Peer country comparison analysis"
        
        # Create the prompt
        prompt = f"""You are a data analyst specializing in developmental outcomes and policy research. 

Context: {context}

Data (JSON format - sampled for analysis):
{reduced_data_json}

Please provide a 3-5 bullet point summary about the key trends shown in this data. Explain these trends within the context of what we know about {display_name} in {country_name if country_name else 'the selected countries'}. Focus on:
1. Key patterns and trends
2. Notable changes over time
3. Contextual insights about what these trends might indicate
4. Do not provide any policy advice or recommendations yourself. 
5. Do not be vague and make sure that you support every argument you make with the data and evidence in front of you.
Keep the response concise, professional, and accessible to policy makers."""
        
        # Prepare metadata for Sessions tracking
        session_metadata = {
            "langfuse_session_id": st.session_state.get('conversation_id', 'visualization_session'),
            "task": "visualization_insights",
            "analysis_type": analysis_type,
            "indicator_name": indicator_name
        }
        
        # Add prompt to metadata if logging is enabled
        from langfuse_config import truncate_prompt
        truncated_prompt = truncate_prompt(prompt)
        if truncated_prompt:
            session_metadata["prompt_text"] = truncated_prompt
            session_metadata["prompt_length"] = len(prompt)
        
        # Make API call with Sessions tracking
        response = client.chat.completions.create(
            name="visualization_insights",
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert data analyst specializing in developmental outcomes and policy research."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500,
            metadata=session_metadata
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def smart_deduplicate_segmentation_data(df, segmentation_columns):
    """
    Smart deduplication for segmentation data.
    For each year, keep the record with the most complete segmentation data.
    """
    if df.empty:
        return df
    
    # Group by year and find the best record for each year
    deduplicated_data = []
    
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        
        if len(year_data) == 1:
            # Only one record for this year, keep it
            deduplicated_data.append(year_data.iloc[0])
        else:
            # Multiple records for this year, choose the best one
            best_record = None
            best_score = -1
            
            for idx, record in year_data.iterrows():
                # Score based on number of non-null segmentation values
                score = 0
                for col in segmentation_columns:
                    if col in record and not pd.isna(record[col]):
                        score += 1
                
                # If this record has a better score, or same score but more recent (higher index)
                if score > best_score or (score == best_score and idx > best_record.name if best_record is not None else True):
                    best_record = record
                    best_score = score
            
            if best_record is not None:
                deduplicated_data.append(best_record)
    
    # Convert back to DataFrame
    if deduplicated_data:
        return pd.DataFrame(deduplicated_data)
    else:
        return df.iloc[0:0]  # Return empty DataFrame with same structure

def format_annotation_value(value, dataset_values):
    """Format annotation value based on dataset characteristics"""
    if pd.isna(value):
        return None  # Skip annotation
    
    # Convert value to numeric if it's a string
    try:
        value = float(value)
    except (ValueError, TypeError):
        return f'{value}'  # Return as string if conversion fails
    
    # Analyze the entire dataset to determine formatting strategy
    valid_values = []
    for v in dataset_values:
        if not pd.isna(v):
            try:
                valid_values.append(float(v))
            except (ValueError, TypeError):
                continue  # Skip non-numeric values
    
    if not valid_values:
        return f'{value:.1f}'  # Default to decimal if no valid data
    
    # Check if all values are whole numbers
    all_whole_numbers = all(v == int(v) for v in valid_values)
    
    if all_whole_numbers:
        return f'{int(value)}'  # No decimal for pure integer datasets
    else:
        return f'{value:.1f}'   # Decimal for mixed or decimal datasets

def create_time_trend_chart(df, selected_country, indicator_name):
    """Create time trend chart for selected country and indicator"""
    if df is None or selected_country is None or indicator_name is None:
        return None
    
    # Get indicator mapping
    indicator = OUTCOME_INDICATORS.get(indicator_name)
    if indicator is None:
        st.warning(f"Unknown indicator: {indicator_name}")
        return None
    
    total_column = indicator["total"]
    display_name = indicator["display_name"]
    unit = indicator["unit"]
    
    # Filter data for selected country
    country_data = df[df['country'] == selected_country].copy()
    
    if country_data.empty:
        st.warning(f"No data available for {selected_country}")
        return None
    
    # Ensure year column is numeric
    country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
    
    # Check if the required column exists and has valid data
    if total_column not in country_data.columns:
        st.warning(f"Column '{total_column}' not found in dataset for {selected_country}")
        return None
    
    # Ensure indicator column is numeric
    country_data[total_column] = pd.to_numeric(country_data[total_column], errors='coerce')
    
    # Remove rows with NaN values for the indicator
    country_data = country_data.dropna(subset=[total_column])
    
    if country_data.empty:
        st.warning(f"No valid data available for {selected_country} and {indicator_name}")
        return None
    
    # Sort by year
    country_data = country_data.sort_values('year')
    
    # Deduplicate by year (keep first occurrence for each year)
    country_data = country_data.drop_duplicates(subset=['year'], keep='first')
    
    # Check if we have multiple years
    years = country_data['year'].unique()
    
    if len(years) == 1:
        # Single year - create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        year = years[0]
        value = country_data[total_column].iloc[0]
        
        bars = ax.bar([f'{selected_country} ({year})'], [value], 
                     color='#667eea', alpha=0.8, edgecolor='#764ba2', linewidth=2)
        
        ax.set_title(f'{display_name}\n{selected_country} - {year}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel(f'{display_name} ({unit})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Country (Year)', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        ax.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Customize the plot
        ax.grid(True, alpha=0.3)
        # Safety check for NaN or infinite values
        if pd.isna(value) or np.isinf(value):
            ax.set_ylim(0, 10)  # Default range if data is invalid
        else:
            ax.set_ylim(0, value + max(value * 0.1, 5))   # adds 10% or 5 units of space

        
    else:
        # Multiple years - create line chart
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(country_data['year'], country_data[total_column], 
                marker='o', linewidth=3, markersize=6, color='#667eea', 
                markerfacecolor='#764ba2', markeredgecolor='white', markeredgewidth=1)
        
        # Calculate year range for title
        valid_data = country_data.dropna(subset=[total_column])
        if not valid_data.empty:
            year_start = int(valid_data['year'].min())
            year_end = int(valid_data['year'].max())
            year_range = f"{year_start}-{year_end}"
        else:
            year_range = "Over Time"
        
        ax.set_title(f'{display_name} ({year_range})\n{selected_country}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{display_name} ({unit})', fontsize=12, fontweight='bold')
        
        # Customize the plot
        ax.grid(True, alpha=0.3)
        # Set x-axis limits based on actual data range (only years with valid data)
        valid_data = country_data.dropna(subset=[total_column])
        if not valid_data.empty:
            min_year = float(valid_data['year'].min())
            max_year = float(valid_data['year'].max())
            ax.set_xlim(min_year - 0.5, max_year + 0.5)
        else:
            # Fallback to full range if no valid data
            min_year = float(country_data['year'].min())
            max_year = float(country_data['year'].max())
            ax.set_xlim(min_year - 0.5, max_year + 0.5)

        
        # Add value labels on points
        data_points = list(zip(country_data['year'], country_data[total_column]))
        dataset_values = country_data[total_column].tolist()  # Get all values for analysis
        
        if len(data_points) >= 11:
            # Annotate every other point starting from the end
            for i in range(len(data_points) - 1, -1, -2):  # Start from end, step by -2
                year, value = data_points[i]
                annotation_text = format_annotation_value(value, dataset_values)
                if annotation_text is not None:
                    ax.annotate(annotation_text, (year, value), 
                               textcoords="offset points", xytext=(0,10), ha='center',
                               fontsize=8, fontweight='bold')
        else:
            # Annotate all points if less than 11
            for year, value in data_points:
                annotation_text = format_annotation_value(value, dataset_values)
                if annotation_text is not None:
                    ax.annotate(annotation_text, (year, value), 
                               textcoords="offset points", xytext=(0,10), ha='center',
                               fontsize=8, fontweight='bold')
        
        ymax = country_data[total_column].max()
        # Safety check for NaN or infinite values
        if pd.isna(ymax) or np.isinf(ymax):
            ax.set_ylim(0, 10)  # Default range if data is invalid
        else:
            ax.set_ylim(0, ymax + max(ymax * 0.1, 5))   # adds 10% or 5 units of space

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig

def create_segmented_trend_chart(df, selected_country, segmentation_type, indicator_name):
    """Create segmented trend chart for selected country and segmentation type"""
    if df is None or selected_country is None or segmentation_type is None or indicator_name is None:
        return None
    
    # Get indicator mapping
    indicator = OUTCOME_INDICATORS.get(indicator_name)
    if indicator is None:
        st.warning(f"Unknown indicator: {indicator_name}")
        return None
    
    # Get segmentation options for this indicator
    segmentation_options = SEGMENTATION_OPTIONS.get(indicator_name, {})
    if segmentation_type not in segmentation_options:
        st.warning(f"Segmentation type '{segmentation_type}' not available for {indicator_name}")
        return None
    
    segmentation_config = segmentation_options[segmentation_type]
    display_name = indicator["display_name"]
    unit = indicator["unit"]
    
    # Filter data for selected country
    country_data = df[df['country'] == selected_country].copy()
    
    if country_data.empty:
        st.warning(f"No data available for {selected_country}")
        return None
    
    # Ensure year column is numeric
    country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
    
    # Get segmentation columns for validation and processing
    columns = {key: val for key, val in segmentation_config.items() if key != 'labels'}
    segmentation_columns = list(columns.values())
    
    # Validate that segmentation columns exist in the dataset
    missing_columns = [col for col in segmentation_columns if col not in country_data.columns]
    if missing_columns:
        st.warning(f"⚠️ Missing segmentation columns in dataset: {missing_columns}")
        st.info("The following segmentation columns are not available in the dataset. Please check your data or select a different segmentation type.")
        return None
    
    # Ensure all indicator columns are numeric
    for column in segmentation_columns:
        if column in country_data.columns:
            country_data[column] = pd.to_numeric(country_data[column], errors='coerce')
    
    # Sort by year
    country_data = country_data.sort_values('year')
    
    # Smart deduplication: keep the record with the most complete segmentation data for each year
    original_count = len(country_data)
    country_data = smart_deduplicate_segmentation_data(country_data, segmentation_columns)
    deduplicated_count = len(country_data)
    
    # Calculate data completeness metrics
    total_years = len(country_data['year'].unique())
    complete_years = 0
    for year in country_data['year'].unique():
        year_data = country_data[country_data['year'] == year]
        if not year_data.empty:
            # Check if all segmentation columns have non-null values for this year
            has_all_data = all(not pd.isna(year_data[col].iloc[0]) for col in segmentation_columns if col in year_data.columns)
            if has_all_data:
                complete_years += 1
    
    # Show data completeness information
    if original_count != deduplicated_count:
        st.info(f"📊 Data Processing: {original_count} records → {deduplicated_count} records (removed {original_count - deduplicated_count} duplicates)")
    
    completeness_pct = (complete_years / total_years * 100) if total_years > 0 else 0
    st.info(f"📈 Data Completeness: {complete_years}/{total_years} years ({completeness_pct:.1f}%) have complete segmentation data")
    
    # Check if we have multiple years
    years = country_data['year'].unique()
    
    # Get the columns and labels for this segmentation
    labels = segmentation_config['labels']
    colors = ['#667eea', '#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(labels)]
    
    if len(years) == 1:
        # Single year - create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 7))
        year = years[0]
        
        # Get data for this year
        year_data = country_data[country_data['year'] == year].iloc[0]
        
        # Get values for each segment
        values = []
        valid_labels = []
        valid_colors = []
        
        for i, (key, column) in enumerate(columns.items()):
            if column in year_data and not pd.isna(year_data[column]):
                values.append(year_data[column])
                valid_labels.append(labels[i])
                valid_colors.append(colors[i])
        
        if not values:
            st.warning(f"No valid data available for {selected_country} in {year}")
            return None
        
        bars = ax.bar(valid_labels, values, color=valid_colors, alpha=0.8, 
                     edgecolor='#2c3e50', linewidth=2)
        
        ax.set_title(f'{display_name} by {segmentation_type}\n{selected_country} - {year}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel(f'{display_name} ({unit})', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{segmentation_type} Category', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Customize the plot
        ax.grid(True, alpha=0.3)
        # Safety check for NaN or infinite values
        if values and not any(pd.isna(v) or np.isinf(v) for v in values):
            ax.set_ylim(0, max(values) + max(max(values) * 0.1, 5))
        else:
            ax.set_ylim(0, 10)  # Default range if data is invalid
        
    else:
        # Multiple years - create line chart with multiple lines
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot lines for each segment
        markers = ['o', 's', '^', 'D', 'v']
        for i, (key, column) in enumerate(columns.items()):
            if column in country_data.columns:
                ax.plot(country_data['year'], country_data[column], 
                       marker=markers[i % len(markers)], linewidth=3, markersize=6, 
                       color=colors[i], markerfacecolor=colors[i], 
                       markeredgecolor='white', markeredgewidth=1,
                       label=labels[i], alpha=0.9)
        
        # Calculate year range for title (using all columns)
        all_years = []
        for column in columns.values():
            if column in country_data.columns:
                valid_data = country_data.dropna(subset=[column])
                if not valid_data.empty:
                    all_years.extend(valid_data['year'].tolist())
        
        if all_years:
            year_start = int(min(all_years))
            year_end = int(max(all_years))
            year_range = f"{year_start}-{year_end}"
        else:
            year_range = "Over Time"
        
        ax.set_title(f'{display_name} by {segmentation_type} ({year_range})\n{selected_country}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{display_name} ({unit})', fontsize=12, fontweight='bold')
        
        # Add legend
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        
        # Customize the plot
        ax.grid(True, alpha=0.3)
        # Set x-axis limits based on actual data range (only years with valid data)
        valid_years = []
        for column in columns.values():
            if column in country_data.columns:
                valid_data = country_data.dropna(subset=[column])
                if not valid_data.empty:
                    valid_years.extend(valid_data['year'].tolist())
        
        if valid_years:
            min_year = float(min(valid_years))
            max_year = float(max(valid_years))
            ax.set_xlim(min_year - 0.5, max_year + 0.5)
        else:
            # Fallback to full range if no valid data
            min_year = float(country_data['year'].min())
            max_year = float(country_data['year'].max())
            ax.set_xlim(min_year - 0.5, max_year + 0.5)
        
        # Add value labels on all points
        data_rows = list(country_data.iterrows())
        # Get all values from all columns for analysis
        all_dataset_values = []
        for column in columns.values():
            if column in country_data.columns:
                all_dataset_values.extend(country_data[column].tolist())
        
        if len(data_rows) >= 11:
            # Annotate every other point starting from the end
            for i in range(len(data_rows) - 1, -1, -2):  # Start from end, step by -2
                year, row = data_rows[i]
                for j, (key, column) in enumerate(columns.items()):
                    if column in row:
                        annotation_text = format_annotation_value(row[column], all_dataset_values)
                        if annotation_text is not None:
                            ax.annotate(annotation_text, 
                                       (row['year'], row[column]), 
                                       textcoords="offset points", xytext=(0,10), ha='center',
                                       fontsize=8, color=colors[j], fontweight='bold')
        else:
            # Annotate all points if less than 11
            for year, row in data_rows:
                for j, (key, column) in enumerate(columns.items()):
                    if column in row:
                        annotation_text = format_annotation_value(row[column], all_dataset_values)
                        if annotation_text is not None:
                            ax.annotate(annotation_text, 
                                       (row['year'], row[column]), 
                                       textcoords="offset points", xytext=(0,10), ha='center',
                                       fontsize=8, color=colors[j], fontweight='bold')
        
        # Set y-axis limits
        all_values = []
        for column in columns.values():
            if column in country_data.columns:
                max_val = country_data[column].max()
                if not pd.isna(max_val) and not np.isinf(max_val):
                    all_values.append(max_val)
        
        if all_values:
            ymax = max(all_values)
            # Safety check for NaN or infinite values
            if pd.isna(ymax) or np.isinf(ymax):
                ax.set_ylim(0, 10)  # Default range if data is invalid
            else:
                ax.set_ylim(0, ymax + max(ymax * 0.1, 5))
        else:
            ax.set_ylim(0, 10)  # Default range if no valid data
        
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig

def create_country_comparison_chart(df, selected_countries, indicator_name):
    """Create country comparison chart for selected countries and indicator"""
    if df is None or not selected_countries or len(selected_countries) < 2:
        return None
    
    indicator_info = OUTCOME_INDICATORS[indicator_name]
    total_column = indicator_info['total']
    display_name = indicator_info['display_name']
    unit = indicator_info['unit']
    
    # Filter data for selected countries
    comparison_data = df[df['country'].isin(selected_countries)].copy()
    
    if comparison_data.empty:
        return None
    
    # Ensure year column is numeric
    comparison_data['year'] = pd.to_numeric(comparison_data['year'], errors='coerce')
    
    # Ensure indicator column is numeric
    comparison_data[total_column] = pd.to_numeric(comparison_data[total_column], errors='coerce')
    
    # Sort by year
    comparison_data = comparison_data.sort_values('year')
    
    # Deduplicate by year and country (keep first occurrence for each year-country combination)
    comparison_data = comparison_data.drop_duplicates(subset=['year', 'country'], keep='first')
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors and markers for countries (limit to 3)
    colors = ['#667eea', '#e74c3c', '#3498db']
    markers = ['o', 's', '^']
    
    # Plot lines for each country
    for i, country in enumerate(selected_countries):
        country_data = comparison_data[comparison_data['country'] == country].sort_values('year')
        if not country_data.empty:
            ax.plot(country_data['year'], country_data[total_column], 
                   marker=markers[i % len(markers)], linewidth=3, markersize=6, 
                   color=colors[i], markerfacecolor=colors[i], 
                   markeredgecolor='white', markeredgewidth=1,
                   label=country, alpha=0.9)
    
    # Calculate year range for title
    all_years = []
    for country in selected_countries:
        country_data = comparison_data[comparison_data['country'] == country]
        valid_data = country_data.dropna(subset=[total_column])
        if not valid_data.empty:
            all_years.extend(valid_data['year'].tolist())
    
    if all_years:
        year_start = int(min(all_years))
        year_end = int(max(all_years))
        year_range = f"{year_start}-{year_end}"
    else:
        year_range = "Over Time"
    
    # Create subtitle with country names
    country_subtitle = " vs. ".join(selected_countries)
    
    ax.set_title(f'{display_name} Comparison ({year_range})\n{country_subtitle}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{display_name} ({unit})', fontsize=12, fontweight='bold')
    
    # Add legend
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # Customize the plot
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits based on actual data range
    if all_years:
        min_year = float(min(all_years))
        max_year = float(max(all_years))
        ax.set_xlim(min_year - 0.5, max_year + 0.5)
    
    # Add value labels on points
    dataset_values = []
    for country in selected_countries:
        country_data = comparison_data[comparison_data['country'] == country]
        dataset_values.extend(country_data[total_column].tolist())
    
    for i, country in enumerate(selected_countries):
        country_data = comparison_data[comparison_data['country'] == country].sort_values('year')
        if not country_data.empty:
            data_points = list(zip(country_data['year'], country_data[total_column]))
            if len(data_points) >= 11:
                # Annotate every other point starting from the end
                for j in range(len(data_points) - 1, -1, -2):
                    year, value = data_points[j]
                    annotation_text = format_annotation_value(value, dataset_values)
                    if annotation_text is not None:
                        ax.annotate(annotation_text, (year, value), 
                                   textcoords="offset points", xytext=(0,10), ha='center',
                                   fontsize=8, fontweight='bold', color=colors[i])
            else:
                # Annotate all points if less than 11
                for year, value in data_points:
                    annotation_text = format_annotation_value(value, dataset_values)
                    if annotation_text is not None:
                        ax.annotate(annotation_text, (year, value), 
                                   textcoords="offset points", xytext=(0,10), ha='center',
                                   fontsize=8, fontweight='bold', color=colors[i])
    
    # Set y-axis limits
    all_values = []
    for country in selected_countries:
        country_data = comparison_data[comparison_data['country'] == country]
        if total_column in country_data.columns:
            max_val = country_data[total_column].max()
            if not pd.isna(max_val) and not np.isinf(max_val):
                all_values.append(max_val)
    
    if all_values:
        ymax = max(all_values)
        if pd.isna(ymax) or np.isinf(ymax):
            ax.set_ylim(0, 10)
        else:
            ax.set_ylim(0, ymax + max(ymax * 0.1, 5))
    else:
        ax.set_ylim(0, 10)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig

def calculate_global_averages(df, indicator_name):
    """Calculate global averages for each year for the given indicator"""
    if df is None or indicator_name is None:
        return None
    
    indicator_info = OUTCOME_INDICATORS[indicator_name]
    total_column = indicator_info['total']
    
    # Filter out rows with NaN values for the indicator
    valid_data = df.dropna(subset=[total_column]).copy()
    
    if valid_data.empty:
        return None
    
    # Ensure year column is numeric
    valid_data['year'] = pd.to_numeric(valid_data['year'], errors='coerce')
    
    # Ensure indicator column is numeric
    valid_data[total_column] = pd.to_numeric(valid_data[total_column], errors='coerce')
    
    # Remove rows with NaN values after conversion
    valid_data = valid_data.dropna(subset=[total_column, 'year'])
    
    if valid_data.empty:
        return None
    
    # Calculate global average by year
    global_averages = valid_data.groupby('year')[total_column].agg(['mean', 'count']).reset_index()
    global_averages.columns = ['year', 'global_average', 'country_count']
    
    # Sort by year
    global_averages = global_averages.sort_values('year')
    
    return global_averages

def calculate_regional_averages(df, indicator_name, selected_country):
    """Calculate regional averages for each year for the given indicator and country"""
    if df is None or indicator_name is None or selected_country is None:
        return None
    
    indicator_info = OUTCOME_INDICATORS[indicator_name]
    total_column = indicator_info['total']
    
    # Get the region of the selected country
    country_data = df[df['country'] == selected_country]
    if country_data.empty:
        return None
    
    region = country_data['region'].iloc[0]
    if pd.isna(region):
        return None
    
    # Filter data for countries in the same region
    regional_data = df[df['region'] == region].copy()
    
    if regional_data.empty:
        return None
    
    # Filter out rows with NaN values for the indicator
    valid_data = regional_data.dropna(subset=[total_column]).copy()
    
    if valid_data.empty:
        return None
    
    # Ensure year column is numeric
    valid_data['year'] = pd.to_numeric(valid_data['year'], errors='coerce')
    
    # Ensure indicator column is numeric
    valid_data[total_column] = pd.to_numeric(valid_data[total_column], errors='coerce')
    
    # Remove rows with NaN values after conversion
    valid_data = valid_data.dropna(subset=[total_column, 'year'])
    
    if valid_data.empty:
        return None
    
    # Calculate regional average by year
    regional_averages = valid_data.groupby('year')[total_column].agg(['mean', 'count']).reset_index()
    regional_averages.columns = ['year', 'regional_average', 'country_count']
    
    # Sort by year
    regional_averages = regional_averages.sort_values('year')
    
    return regional_averages, region

def create_global_benchmarking_chart(df, selected_country, indicator_name):
    """Create global benchmarking chart comparing selected country to global average"""
    if df is None or selected_country is None or indicator_name is None:
        return None
    
    # Get indicator mapping
    indicator = OUTCOME_INDICATORS.get(indicator_name)
    if indicator is None:
        st.warning(f"Unknown indicator: {indicator_name}")
        return None
    
    total_column = indicator["total"]
    display_name = indicator["display_name"]
    unit = indicator["unit"]
    
    # Filter data for selected country
    country_data = df[df['country'] == selected_country].copy()
    
    if country_data.empty:
        st.warning(f"No data available for {selected_country}")
        return None
    
    # Ensure year column is numeric
    country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
    
    # Ensure indicator column is numeric
    country_data[total_column] = pd.to_numeric(country_data[total_column], errors='coerce')
    
    # Remove rows with NaN values for the indicator
    country_data = country_data.dropna(subset=[total_column])
    
    if country_data.empty:
        st.warning(f"No valid data available for {selected_country} and {indicator_name}")
        return None
    
    # Calculate global averages
    global_averages = calculate_global_averages(df, indicator_name)
    
    if global_averages is None or global_averages.empty:
        st.warning("Could not calculate global averages")
        return None
    
    # Calculate regional averages
    regional_result = calculate_regional_averages(df, indicator_name, selected_country)
    if regional_result is None:
        regional_averages = None
        region_name = None
    else:
        regional_averages, region_name = regional_result
    
    # Sort by year
    country_data = country_data.sort_values('year')
    
    # Deduplicate by year (keep first occurrence for each year)
    country_data = country_data.drop_duplicates(subset=['year'], keep='first')
    
    # Filter averages to only include years where the country has data
    country_years = set(country_data['year'].unique())
    global_averages = global_averages[global_averages['year'].isin(country_years)]
    
    if regional_averages is not None:
        regional_averages = regional_averages[regional_averages['year'].isin(country_years)]
    
    if global_averages.empty:
        st.warning("No global averages available for the years where this country has data")
        return None
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot country data
    ax.plot(country_data['year'], country_data[total_column], 
            marker='o', linewidth=3, markersize=6, color='#667eea', 
            markerfacecolor='#667eea', markeredgecolor='white', markeredgewidth=1,
            label=f'{selected_country}', alpha=0.9)
    
    # Plot global average
    ax.plot(global_averages['year'], global_averages['global_average'], 
            marker='s', linewidth=3, markersize=6, color='#e74c3c', 
            markerfacecolor='#e74c3c', markeredgecolor='white', markeredgewidth=1,
            label='Global Average', alpha=0.9, linestyle='--')
    
    # Plot regional average if available
    if regional_averages is not None and not regional_averages.empty:
        ax.plot(regional_averages['year'], regional_averages['regional_average'], 
                marker='^', linewidth=3, markersize=6, color='#2ecc71', 
                markerfacecolor='#2ecc71', markeredgecolor='white', markeredgewidth=1,
                label=f'{region_name} Average', alpha=0.9, linestyle=':')
    
    # Calculate year range for title (now both datasets have the same years)
    if not country_data.empty and not global_averages.empty:
        year_start = int(min(country_data['year'].unique()))
        year_end = int(max(country_data['year'].unique()))
        year_range = f"{year_start}-{year_end}"
    else:
        year_range = "Over Time"
    
    # Create title with region name if available
    if region_name:
        title = f'{display_name} vs {region_name} Average vs Global Average ({year_range})\n{selected_country}'
    else:
        title = f'{display_name} vs Global Average ({year_range})\n{selected_country}'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{display_name} ({unit})', fontsize=12, fontweight='bold')
    
    # Add legend
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # Customize the plot
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits based on actual data range (both datasets now have the same years)
    if not country_data.empty:
        min_year = float(country_data['year'].min())
        max_year = float(country_data['year'].max())
        ax.set_xlim(min_year - 0.5, max_year + 0.5)
    
    # Add value labels on points (every other point if too many)
    country_data_points = list(zip(country_data['year'], country_data[total_column]))
    global_data_points = list(zip(global_averages['year'], global_averages['global_average']))
    
    # Combine all values for annotation formatting
    all_dataset_values = country_data[total_column].tolist() + global_averages['global_average'].tolist()
    
    # Add regional data points if available
    if regional_averages is not None and not regional_averages.empty:
        regional_data_points = list(zip(regional_averages['year'], regional_averages['regional_average']))
        all_dataset_values.extend(regional_averages['regional_average'].tolist())
    else:
        regional_data_points = []
    
    # Annotate country data points
    if len(country_data_points) >= 11:
        # Annotate every other point starting from the end
        for i in range(len(country_data_points) - 1, -1, -2):
            year, value = country_data_points[i]
            annotation_text = format_annotation_value(value, all_dataset_values)
            if annotation_text is not None:
                ax.annotate(annotation_text, (year, value), 
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=8, fontweight='bold', color='#667eea')
    else:
        # Annotate all points if less than 11
        for year, value in country_data_points:
            annotation_text = format_annotation_value(value, all_dataset_values)
            if annotation_text is not None:
                ax.annotate(annotation_text, (year, value), 
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=8, fontweight='bold', color='#667eea')
    
    # Annotate global average points (fewer annotations to avoid clutter)
    if len(global_data_points) >= 8:
        # Annotate every third point
        for i in range(len(global_data_points) - 1, -1, -3):
            year, value = global_data_points[i]
            annotation_text = format_annotation_value(value, all_dataset_values)
            if annotation_text is not None:
                ax.annotate(annotation_text, (year, value), 
                           textcoords="offset points", xytext=(0,-15), ha='center',
                           fontsize=8, fontweight='bold', color='#e74c3c')
    else:
        # Annotate all points if less than 8
        for year, value in global_data_points:
            annotation_text = format_annotation_value(value, all_dataset_values)
            if annotation_text is not None:
                ax.annotate(annotation_text, (year, value), 
                           textcoords="offset points", xytext=(0,-15), ha='center',
                           fontsize=8, fontweight='bold', color='#e74c3c')
    
    # Annotate regional average points if available
    if regional_data_points:
        if len(regional_data_points) >= 6:
            # Annotate every other point
            for i in range(len(regional_data_points) - 1, -1, -2):
                year, value = regional_data_points[i]
                annotation_text = format_annotation_value(value, all_dataset_values)
                if annotation_text is not None:
                    ax.annotate(annotation_text, (year, value), 
                               textcoords="offset points", xytext=(0,15), ha='center',
                               fontsize=8, fontweight='bold', color='#2ecc71')
        else:
            # Annotate all points if less than 6
            for year, value in regional_data_points:
                annotation_text = format_annotation_value(value, all_dataset_values)
                if annotation_text is not None:
                    ax.annotate(annotation_text, (year, value), 
                               textcoords="offset points", xytext=(0,15), ha='center',
                               fontsize=8, fontweight='bold', color='#2ecc71')
    
    # Set y-axis limits
    all_values = country_data[total_column].tolist() + global_averages['global_average'].tolist()
    if regional_averages is not None and not regional_averages.empty:
        all_values.extend(regional_averages['regional_average'].tolist())
    
    if all_values:
        ymax = max(all_values)
        if pd.isna(ymax) or np.isinf(ymax):
            ax.set_ylim(0, 10)
        else:
            ax.set_ylim(0, ymax + max(ymax * 0.1, 5))
    else:
        ax.set_ylim(0, 10)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig

def create_benchmarking_bar_chart(df, selected_country, indicator_name):
    """Create bar chart comparing country vs global average for last 3 years"""
    if df is None or selected_country is None or indicator_name is None:
        return None
    
    # Get indicator mapping
    indicator = OUTCOME_INDICATORS.get(indicator_name)
    if indicator is None:
        st.warning(f"Unknown indicator: {indicator_name}")
        return None
    
    total_column = indicator["total"]
    display_name = indicator["display_name"]
    unit = indicator["unit"]
    
    # Filter data for selected country
    country_data = df[df['country'] == selected_country].copy()
    
    if country_data.empty:
        st.warning(f"No data available for {selected_country}")
        return None
    
    # Ensure year and indicator columns are numeric
    country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
    country_data[total_column] = pd.to_numeric(country_data[total_column], errors='coerce')
    
    # Remove rows with NaN values for the indicator
    country_data = country_data.dropna(subset=[total_column])
    
    if country_data.empty:
        st.warning(f"No valid data available for {selected_country} and {indicator_name}")
        return None
    
    # Sort by year and get last 3 years
    country_data = country_data.sort_values('year')
    
    # Deduplicate by year (keep first occurrence for each year)
    country_data = country_data.drop_duplicates(subset=['year'], keep='first')
    
    last_3_years_data = country_data.tail(3)
    
    if last_3_years_data.empty:
        st.warning("No data available for the last 3 years")
        return None
    
    # Calculate global averages for the same years
    global_averages = calculate_global_averages(df, indicator_name)
    
    if global_averages is None or global_averages.empty:
        st.warning("Could not calculate global averages")
        return None
    
    # Calculate regional averages for the same years
    regional_result = calculate_regional_averages(df, indicator_name, selected_country)
    if regional_result is None:
        regional_averages = None
        region_name = None
    else:
        regional_averages, region_name = regional_result
    
    # Ensure year columns are numeric
    global_averages['year'] = pd.to_numeric(global_averages['year'], errors='coerce')
    if regional_averages is not None:
        regional_averages['year'] = pd.to_numeric(regional_averages['year'], errors='coerce')
    
    # Filter averages to only include the last 3 years
    last_3_years = last_3_years_data['year'].unique()
    global_averages_filtered = global_averages[global_averages['year'].isin(last_3_years)]
    
    if regional_averages is not None:
        regional_averages_filtered = regional_averages[regional_averages['year'].isin(last_3_years)]
    else:
        regional_averages_filtered = None
    
    if global_averages_filtered.empty:
        st.warning("No global averages available for the last 3 years")
        return None
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting
    years = sorted(last_3_years_data['year'].unique())
    country_values = []
    global_values = []
    regional_values = []
    
    for year in years:
        country_value = last_3_years_data[last_3_years_data['year'] == year][total_column].iloc[0]
        global_value = global_averages_filtered[global_averages_filtered['year'] == year]['global_average'].iloc[0]
        
        country_values.append(country_value)
        global_values.append(global_value)
        
        # Add regional value if available
        if regional_averages_filtered is not None and not regional_averages_filtered.empty:
            regional_year_data = regional_averages_filtered[regional_averages_filtered['year'] == year]
            if not regional_year_data.empty:
                regional_value = regional_year_data['regional_average'].iloc[0]
                regional_values.append(regional_value)
            else:
                regional_values.append(None)
        else:
            regional_values.append(None)
    
    # Set up bar positions
    x = np.arange(len(years))
    width = 0.25  # Reduced width to accommodate three bars
    
    # Create bars
    bars1 = ax.bar(x - width, country_values, width, label=f'{selected_country}', 
                   color='#667eea', alpha=0.8, edgecolor='#2c3e50', linewidth=1)
    
    # Create regional bars if data is available
    bars2 = None
    if any(v is not None for v in regional_values):
        regional_label = f'{region_name} Average' if region_name else 'Regional Average'
        bars2 = ax.bar(x, regional_values, width, label=regional_label, 
                       color='#2ecc71', alpha=0.8, edgecolor='#2c3e50', linewidth=1)
    
    bars3 = ax.bar(x + width, global_values, width, label='Global Average', 
                   color='#e74c3c', alpha=0.8, edgecolor='#2c3e50', linewidth=1)
    
    # Customize the chart
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{display_name} ({unit})', fontsize=12, fontweight='bold')
    # Create dynamic title based on available data
    if region_name and any(v is not None for v in regional_values):
        title = f'{display_name} Comparison\n{selected_country} vs {region_name} Average vs Global Average'
    else:
        title = f'{display_name} Comparison\n{selected_country} vs Global Average'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([int(year) for year in years])
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    all_values_for_max = country_values + global_values
    if any(v is not None for v in regional_values):
        all_values_for_max.extend([v for v in regional_values if v is not None])
    
    for bar, value in zip(bars1, country_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(all_values_for_max) * 0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add regional value labels if available
    if bars2 is not None:
        for bar, value in zip(bars2, regional_values):
            if value is not None:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(all_values_for_max) * 0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, value in zip(bars3, global_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(all_values_for_max) * 0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Set y-axis limits
    all_values = country_values + global_values
    if any(v is not None for v in regional_values):
        all_values.extend([v for v in regional_values if v is not None])
    
    if all_values:
        ymax = max(all_values)
        ax.set_ylim(0, ymax + max(ymax * 0.15, 5))
    
    plt.tight_layout()
    return fig

def create_regional_comparison_chart(df, selected_country, indicator_name):
    """Create horizontal bar chart comparing selected country to all countries in its region for latest year"""
    if df is None or selected_country is None or indicator_name is None:
        return None
    
    # Get indicator mapping
    indicator = OUTCOME_INDICATORS.get(indicator_name)
    if indicator is None:
        st.warning(f"Unknown indicator: {indicator_name}")
        return None
    
    total_column = indicator["total"]
    display_name = indicator["display_name"]
    unit = indicator["unit"]
    
    # Filter data for selected country
    country_data = df[df['country'] == selected_country].copy()
    
    if country_data.empty:
        st.warning(f"No data available for {selected_country}")
        return None
    
    # Ensure year and indicator columns are numeric
    country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
    country_data[total_column] = pd.to_numeric(country_data[total_column], errors='coerce')
    
    # Remove rows with NaN values for the indicator
    country_data = country_data.dropna(subset=[total_column])
    
    if country_data.empty:
        st.warning(f"No valid data available for {selected_country} and {indicator_name}")
        return None
    
    # Get the latest year with data for the selected country
    latest_year = country_data['year'].max()
    
    # Get the region of the selected country
    region = country_data['region'].iloc[0]
    if pd.isna(region):
        st.warning(f"No region information available for {selected_country}")
        return None
    
    # Filter data for all countries in the same region for the latest year
    regional_data = df[df['region'] == region].copy()
    regional_data['year'] = pd.to_numeric(regional_data['year'], errors='coerce')
    regional_data[total_column] = pd.to_numeric(regional_data[total_column], errors='coerce')
    
    # Get data for the latest year only
    latest_year_data = regional_data[
        (regional_data['year'] == latest_year) & 
        (regional_data[total_column].notna())
    ].copy()
    
    if latest_year_data.empty:
        st.warning(f"No regional data available for {latest_year}")
        return None
    
    # Sort by value (highest to lowest)
    latest_year_data = latest_year_data.sort_values(total_column, ascending=False)
    
    # Create the horizontal bar chart
    fig, ax = plt.subplots(figsize=(14, max(8, len(latest_year_data) * 0.4)))
    
    # Prepare data for plotting
    countries = latest_year_data['country'].tolist()
    values = latest_year_data[total_column].tolist()
    
    # Create color list (red for selected country, blue for others)
    colors = ['#e74c3c' if country == selected_country else '#667eea' for country in countries]
    
    # Create horizontal bars
    y_pos = np.arange(len(countries))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=1)
    
    # Customize the chart
    ax.set_xlabel(f'{display_name} ({unit})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Country', fontsize=12, fontweight='bold')
    ax.set_title(f'Regional Comparison - {region} - {int(latest_year)}\n{display_name}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(countries)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label=f'{selected_country}'),
        Patch(facecolor='#667eea', label='Other Regional Countries')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.9)
    
    # Customize the plot
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(values) + max(values) * 0.15)
    
    # Invert y-axis to show highest values at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig

def get_country_color_mapping(countries_with_data, selected_country):
    """Create consistent color mapping for countries"""
    # Define color palette
    color_palette = ['#e74c3c', '#667eea', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '8']
    
    # Create color mapping
    color_mapping = {}
    marker_mapping = {}
    
    for i, country in enumerate(countries_with_data):
        color_mapping[country] = color_palette[i % len(color_palette)]
        marker_mapping[country] = markers[i % len(markers)]
    
    return color_mapping, marker_mapping

def create_peer_country_comparison_chart(df, selected_country, indicator_name):
    """Create peer country comparison chart for selected country and its peer countries"""
    if df is None or selected_country is None or indicator_name is None:
        return None
    
    # Get indicator mapping
    indicator = OUTCOME_INDICATORS.get(indicator_name)
    if indicator is None:
        st.warning(f"Unknown indicator: {indicator_name}")
        return None
    
    total_column = indicator["total"]
    display_name = indicator["display_name"]
    unit = indicator["unit"]
    
    # Get peer countries for the selected country
    country_data = df[df['country'] == selected_country]
    if country_data.empty:
        st.warning(f"No data available for {selected_country}")
        return None
    
    # Get peer countries
    peer_countries = []
    explanation = ""
    
    if 'Country 2' in country_data.columns and not pd.isna(country_data['Country 2'].iloc[0]):
        peer_countries.append(country_data['Country 2'].iloc[0])
    if 'Country 3' in country_data.columns and not pd.isna(country_data['Country 3'].iloc[0]):
        peer_countries.append(country_data['Country 3'].iloc[0])
    if 'Country 4' in country_data.columns and not pd.isna(country_data['Country 4'].iloc[0]):
        peer_countries.append(country_data['Country 4'].iloc[0])
    
    if 'Explanation' in country_data.columns and not pd.isna(country_data['Explanation'].iloc[0]):
        explanation = country_data['Explanation'].iloc[0]
    
    if not peer_countries:
        st.warning(f"No peer countries available for {selected_country}")
        return None, "", []
    
    # Create list of all countries to compare (selected + peers)
    all_countries = [selected_country] + peer_countries
    
    # Filter data for all countries
    comparison_data = df[df['country'].isin(all_countries)].copy()
    
    if comparison_data.empty:
        return None, explanation, []
    
    # Ensure year column is numeric
    comparison_data['year'] = pd.to_numeric(comparison_data['year'], errors='coerce')
    
    # Ensure indicator column is numeric
    comparison_data[total_column] = pd.to_numeric(comparison_data[total_column], errors='coerce')
    
    # Sort by year
    comparison_data = comparison_data.sort_values('year')
    
    # Deduplicate by year and country (keep first occurrence for each year-country combination)
    comparison_data = comparison_data.drop_duplicates(subset=['year', 'country'], keep='first')
    
    # Filter out countries that don't have any valid data for this indicator
    countries_with_data = []
    for country in all_countries:
        country_data = comparison_data[comparison_data['country'] == country]
        valid_data = country_data.dropna(subset=[total_column])
        if not valid_data.empty:
            countries_with_data.append(country)
    
    if not countries_with_data:
        st.warning("No countries have valid data for this indicator")
        return None, explanation, []
    
    # Ensure selected country is always first and has data
    if selected_country not in countries_with_data:
        st.warning(f"No valid data available for {selected_country}")
        return None, explanation, []
    
    # Reorder countries_with_data to put selected country first
    countries_with_data = [selected_country] + [c for c in countries_with_data if c != selected_country]
    
    # Get consistent color and marker mapping
    color_mapping, marker_mapping = get_country_color_mapping(countries_with_data, selected_country)
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot lines for each country using consistent color mapping
    for country in countries_with_data:
        country_data = comparison_data[comparison_data['country'] == country].sort_values('year')
        if not country_data.empty:
            # Make selected country line thicker and more prominent
            linewidth = 4 if country == selected_country else 3
            markersize = 8 if country == selected_country else 6
            alpha = 1.0 if country == selected_country else 0.9
            
            ax.plot(country_data['year'], country_data[total_column], 
                   marker=marker_mapping[country], linewidth=linewidth, markersize=markersize, 
                   color=color_mapping[country], markerfacecolor=color_mapping[country], 
                   markeredgecolor='white', markeredgewidth=1,
                   label=country, alpha=alpha)
    
    # Calculate year range for title
    all_years = []
    for country in countries_with_data:
        country_data = comparison_data[comparison_data['country'] == country]
        valid_data = country_data.dropna(subset=[total_column])
        if not valid_data.empty:
            all_years.extend(valid_data['year'].tolist())
    
    if all_years:
        year_start = int(min(all_years))
        year_end = int(max(all_years))
        year_range = f"{year_start}-{year_end}"
    else:
        year_range = "Over Time"
    
    # Create subtitle with only countries that have data
    peer_countries_with_data = [c for c in countries_with_data if c != selected_country]
    if peer_countries_with_data:
        peer_subtitle = f"{selected_country} vs. {', '.join(peer_countries_with_data)}"
    else:
        peer_subtitle = f"{selected_country} (no peer countries with data)"
    
    ax.set_title(f'{display_name} - Peer Country Comparison ({year_range})\n{peer_subtitle}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{display_name} ({unit})', fontsize=12, fontweight='bold')
    
    # Add legend
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # Customize the plot
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits based on actual data range
    if all_years:
        min_year = float(min(all_years))
        max_year = float(max(all_years))
        ax.set_xlim(min_year - 0.5, max_year + 0.5)
    
    # Add value labels on points
    dataset_values = []
    for country in countries_with_data:
        country_data = comparison_data[comparison_data['country'] == country]
        dataset_values.extend(country_data[total_column].tolist())
    
    for country in countries_with_data:
        country_data = comparison_data[comparison_data['country'] == country].sort_values('year')
        if not country_data.empty:
            data_points = list(zip(country_data['year'], country_data[total_column]))
            if len(data_points) >= 11:
                # Annotate every other point starting from the end
                for j in range(len(data_points) - 1, -1, -2):
                    year, value = data_points[j]
                    annotation_text = format_annotation_value(value, dataset_values)
                    if annotation_text is not None:
                        ax.annotate(annotation_text, (year, value), 
                                   textcoords="offset points", xytext=(0,10), ha='center',
                                   fontsize=8, fontweight='bold', color=color_mapping[country])
            else:
                # Annotate all points if less than 11
                for year, value in data_points:
                    annotation_text = format_annotation_value(value, dataset_values)
                    if annotation_text is not None:
                        ax.annotate(annotation_text, (year, value), 
                                   textcoords="offset points", xytext=(0,10), ha='center',
                                   fontsize=8, fontweight='bold', color=color_mapping[country])
    
    # Set y-axis limits
    all_values = []
    for country in countries_with_data:
        country_data = comparison_data[comparison_data['country'] == country]
        if total_column in country_data.columns:
            max_val = country_data[total_column].max()
            if not pd.isna(max_val) and not np.isinf(max_val):
                all_values.append(max_val)
    
    if all_values:
        ymax = max(all_values)
        if pd.isna(ymax) or np.isinf(ymax):
            ax.set_ylim(0, 10)
        else:
            ax.set_ylim(0, ymax + max(ymax * 0.1, 5))
    else:
        ax.set_ylim(0, 10)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig, explanation, countries_with_data

def create_peer_country_bar_chart(df, selected_country, indicator_name, countries_with_data):
    """Create bar chart for latest year with data for all countries"""
    if df is None or selected_country is None or indicator_name is None or not countries_with_data:
        return None
    
    # Get indicator mapping
    indicator = OUTCOME_INDICATORS.get(indicator_name)
    if indicator is None:
        return None
    
    total_column = indicator["total"]
    display_name = indicator["display_name"]
    unit = indicator["unit"]
    
    # Get consistent color mapping (same as line chart)
    color_mapping, _ = get_country_color_mapping(countries_with_data, selected_country)
    
    # Filter data for countries with data
    comparison_data = df[df['country'].isin(countries_with_data)].copy()
    
    # Ensure year and indicator columns are numeric
    comparison_data['year'] = pd.to_numeric(comparison_data['year'], errors='coerce')
    comparison_data[total_column] = pd.to_numeric(comparison_data[total_column], errors='coerce')
    
    # Remove rows with NaN values for the indicator
    comparison_data = comparison_data.dropna(subset=[total_column])
    
    # Deduplicate by year and country (keep first occurrence for each year-country combination)
    comparison_data = comparison_data.drop_duplicates(subset=['year', 'country'], keep='first')
    
    if comparison_data.empty:
        return None
    
    # Find the latest year where ALL countries have data
    # Get the latest year for each country
    country_latest_years = {}
    for country in countries_with_data:
        country_data = comparison_data[comparison_data['country'] == country]
        if not country_data.empty:
            country_latest_years[country] = country_data['year'].max()
    
    if not country_latest_years:
        return None
    
    # Find the latest year where all countries have data
    # We need to find a year where ALL countries have data
    all_years = sorted(comparison_data['year'].unique(), reverse=True)  # Start from latest
    
    latest_common_year = None
    for year in all_years:
        year_data = comparison_data[comparison_data['year'] == year]
        countries_in_year = year_data['country'].unique()
        
        # Check if all countries_with_data are present in this year
        if all(country in countries_in_year for country in countries_with_data):
            latest_common_year = year
            break
    
    if latest_common_year is None:
        return None
    
    # Get data for the latest common year
    latest_year_data = comparison_data[comparison_data['year'] == latest_common_year].copy()
    
    if latest_year_data.empty:
        return None
    
    # Sort by value (highest to lowest) - this will naturally put highest values first
    latest_year_data = latest_year_data.sort_values(total_column, ascending=False)
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for plotting
    countries = latest_year_data['country'].tolist()
    values = latest_year_data[total_column].tolist()
    
    # Use consistent color mapping from line chart
    bar_colors = [color_mapping[country] for country in countries]
    
    # Create bars
    bars = ax.bar(countries, values, color=bar_colors, alpha=0.8, 
                 edgecolor='#2c3e50', linewidth=2)
    
    # Make selected country bar thicker/bolder
    for i, country in enumerate(countries):
        if country == selected_country:
            bars[i].set_linewidth(3)
            break
    
    # Customize the chart
    ax.set_xlabel('Country', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{display_name} ({unit})', fontsize=12, fontweight='bold')
    ax.set_title(f'{display_name} - {selected_country} and Peer Countries\n({int(latest_common_year)})', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize the plot
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(values) + max(values) * 0.15)
    
    plt.tight_layout()
    return fig

def main():
    # Initialize session state
    if 'show_time_trends' not in st.session_state:
        st.session_state.show_time_trends = False
    if 'show_segmented_trends' not in st.session_state:
        st.session_state.show_segmented_trends = False
    if 'show_country_comparison' not in st.session_state:
        st.session_state.show_country_comparison = False
    if 'show_global_benchmarking' not in st.session_state:
        st.session_state.show_global_benchmarking = False
    if 'show_peer_benchmarking' not in st.session_state:
        st.session_state.show_peer_benchmarking = False
    if 'selected_indicator' not in st.session_state:
        st.session_state.selected_indicator = "Unemployment"
    
    # Load data
    df = load_data()
    dataset_summaries = load_dataset_summaries()
    
    # Main title section
    st.markdown("""
    <div class="main-title">
        <h1>📊 Visualization Dashboard</h1>
        <p>Outcome Development of Interest: Unemployment: Youth not in Education or Training</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Indicator selection
    st.markdown("### 📊 Select Development Outcome Indicator")
    selected_indicator = st.selectbox(
        "Choose the outcome indicator to analyze:",
        options=list(OUTCOME_INDICATORS.keys()),
        index=list(OUTCOME_INDICATORS.keys()).index(st.session_state.selected_indicator),
        help="Select the development outcome you want to visualize"
    )
    
    # Update session state
    st.session_state.selected_indicator = selected_indicator
    
    # Show selected indicator info
    indicator_info = OUTCOME_INDICATORS[selected_indicator]
    st.info(f"**Selected:** {indicator_info['display_name']} ({indicator_info['unit']})")
    
    # Introduction text
    st.markdown("""
    ### Welcome to the Policy CoPilot Visualization Dashboard
    
    This dashboard provides comprehensive visualizations for analyzing development outcomes across different countries. 
    Select one of the visualization options below to explore the data.
    """)
    
    # Check if we should show time trends
    if st.session_state.show_time_trends:
        # Show Time Trends interface
        st.markdown("### 📊 Country Specific Time-Trend Analysis")
        
        if df is not None:
            # Get countries that have data for the selected indicator
            countries = get_countries_with_data(df, st.session_state.selected_indicator)
            
            if countries:
                # Country selection dropdown
                selected_country = st.selectbox(
                    "Select a Country:",
                    options=countries,
                    index=0,
                    help=f"Choose a country to view its {st.session_state.selected_indicator.lower()} trends over time"
                )
                
                if selected_country:
                    # Show Information on Dataset
                    if st.session_state.selected_indicator in dataset_summaries:
                        with st.expander("📋 Information on Dataset", expanded=True):
                            st.text(dataset_summaries[st.session_state.selected_indicator])
                    
                    # Create and display the chart
                    chart = create_time_trend_chart(df, selected_country, st.session_state.selected_indicator)
                    if chart:
                        st.pyplot(chart)
                        
                        # Generate Insights button
                        st.markdown("---")
                        if st.button("🤖 Generate Insights", key="insights_time_trend", type="secondary"):
                            with st.spinner("Generating insights..."):
                                # Prepare data for AI analysis
                                country_data = df[df['country'] == selected_country].copy()
                                # Ensure year column is numeric before sorting
                                country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
                                country_data = country_data.sort_values('year')
                                indicator_info = OUTCOME_INDICATORS[st.session_state.selected_indicator]
                                total_column = indicator_info['total']
                                
                                # Filter data for the specific indicator
                                chart_data = country_data[['year', total_column]].dropna()
                                data_json = chart_data.to_json(orient='records')
                                
                                # Generate insights
                                insights = generate_insights(
                                    data_json=data_json,
                                    indicator_name=st.session_state.selected_indicator,
                                    country_name=selected_country,
                                    analysis_type="time_trend"
                                )
                                
                                # Display insights
                                st.markdown("### 🤖 AI-Generated Insights")
                                st.info(insights)
                        
                        # Show data summary
                        country_data = df[df['country'] == selected_country].copy()
                        # Ensure year column is numeric before sorting
                        country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
                        country_data = country_data.sort_values('year')
                        st.markdown("### 📈 Data Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            indicator_info = OUTCOME_INDICATORS[st.session_state.selected_indicator]
                            total_column = indicator_info['total']
                            
                            # Get years with valid data for the specific indicator
                            valid_data = country_data.dropna(subset=[total_column])
                            years_with_data = len(valid_data['year'].unique()) if not valid_data.empty else 0
                            st.metric("Years of Data", years_with_data)
                        with col2:
                            unit = indicator_info['unit']
                            
                            # Get the latest year with valid data
                            if not valid_data.empty:
                                latest_data_row = valid_data.iloc[-1]
                                latest_year = latest_data_row['year']
                                latest_value = latest_data_row[total_column]
                                
                                # Convert to numeric if needed
                                try:
                                    latest_value = float(latest_value)
                                    st.metric(f"Latest Rate ({latest_year})", f"{latest_value:.1f}{unit}")
                                except (ValueError, TypeError):
                                    st.metric(f"Latest Rate ({latest_year})", f"{latest_value}{unit}")
                            else:
                                st.metric("Latest Rate", "No valid data")
                        with col3:
                            # Calculate trend between first and last valid data points
                            valid_data = country_data.dropna(subset=[total_column])
                            if len(valid_data) > 1:
                                first_value = valid_data[total_column].iloc[0]
                                last_value = valid_data[total_column].iloc[-1]
                                
                                # Convert to numeric if needed
                                try:
                                    first_value = float(first_value)
                                    last_value = float(last_value)
                                    trend = last_value - first_value
                                    year_start = int(valid_data['year'].min())
                                    year_end = int(valid_data['year'].max())
                                    st.metric(f"Overall Trend ({year_start}-{year_end})", f"{trend:+.1f}{unit}")
                                except (ValueError, TypeError):
                                    st.metric("Overall Trend", "N/A")
                            else:
                                st.metric("Overall Trend", "N/A")
                        
                        # Show raw data
                        with st.expander("View Raw Data"):
                            display_name = indicator_info['display_name']
                            # Remove duplicates by year, keeping the first occurrence, and filter out NA values
                            unique_data = country_data[['year', total_column]].drop_duplicates(subset=['year'], keep='first')
                            # Filter out rows where the value column is NA
                            filtered_data = unique_data.dropna(subset=[total_column])
                            st.dataframe(filtered_data.rename(columns={
                                total_column: f'{display_name} ({unit})'
                            }))
            else:
                st.warning(f"⚠️ No countries have data available for {st.session_state.selected_indicator}")
                st.info("Please select a different development outcome indicator.")
        else:
            st.warning("Could not load the dataset. Please check the file path.")
        
        # Back button
        if st.button("← Back to Main Dashboard"):
            st.session_state.show_time_trends = False
            st.rerun()
    
    # Check if we should show segmented trends
    elif st.session_state.show_segmented_trends:
        # Show Segmented Trends interface
        st.markdown("### 🔍 Country Specific Time-Trends Segmented Analysis")
        
        if df is not None:
            # Get countries that have data for the selected indicator
            countries = get_countries_with_data(df, st.session_state.selected_indicator)
            
            if countries:
                # Check if segmentation is available for current indicator
                segmentation_options = SEGMENTATION_OPTIONS.get(st.session_state.selected_indicator, {})
                
                if not segmentation_options:
                    st.warning(f"⚠️ Segmentation is not available for {st.session_state.selected_indicator}")
                    st.info("This indicator does not have segmentation data available.")
                    
                    # Back button
                    if st.button("← Back to Main Dashboard", key="back_segmented_1"):
                        st.session_state.show_segmented_trends = False
                else:
                    # Create two columns for dropdowns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Country selection dropdown
                        selected_country = st.selectbox(
                            "Select a Country:",
                            options=countries,
                            index=0,
                            help=f"Choose a country to view its {st.session_state.selected_indicator.lower()} segmented trends",
                            key="segmented_country"
                        )
                    
                    with col2:
                        # Segmentation type dropdown
                        segmentation_types = list(segmentation_options.keys())
                        segmentation_type = st.selectbox(
                            "Select Segmentation Type:",
                            options=segmentation_types,
                            index=0,
                            help="Choose the segmentation dimension for analysis",
                            key="segmentation_type"
                        )
                    
                    # Only show chart and data after both selections are made
                    if selected_country and segmentation_type:
                        # Show Information on Dataset
                        if st.session_state.selected_indicator in dataset_summaries:
                            with st.expander("📋 Information on Dataset", expanded=True):
                                st.text(dataset_summaries[st.session_state.selected_indicator])
                        
                        st.markdown("---")
                        st.markdown("### 📊 Visualization")
                        
                        # Create and display the chart
                        chart = create_segmented_trend_chart(df, selected_country, segmentation_type, st.session_state.selected_indicator)
                        if chart:
                            st.pyplot(chart)
                            
                            # Generate Insights button
                            st.markdown("---")
                            if st.button("🤖 Generate Insights", key="insights_segmented", type="secondary"):
                                with st.spinner("Generating insights..."):
                                    # Prepare data for AI analysis
                                    country_data = df[df['country'] == selected_country].copy()
                                    # Ensure year column is numeric before sorting
                                    country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
                                    country_data = country_data.sort_values('year')
                                    indicator_info = OUTCOME_INDICATORS[st.session_state.selected_indicator]
                                    segmentation_config = segmentation_options[segmentation_type]
                                    
                                    # Get all segmentation columns
                                    columns = {key: val for key, val in segmentation_config.items() if key != 'labels'}
                                    columns_to_show = ['year'] + list(columns.values())
                                    
                                    # Filter data for the specific indicator and segmentation
                                    chart_data = country_data[columns_to_show].dropna(subset=list(columns.values()), how='all')
                                    data_json = chart_data.to_json(orient='records')
                                    
                                    # Generate insights
                                    insights = generate_insights(
                                        data_json=data_json,
                                        indicator_name=st.session_state.selected_indicator,
                                        country_name=selected_country,
                                        analysis_type="segmented"
                                    )
                                    
                                    # Display insights
                                    st.markdown("### 🤖 AI-Generated Insights")
                                    st.info(insights)
                            
                            # Show raw data (before processing)
                            with st.expander("View Raw Data"):
                                # Get segmentation config and columns
                                segmentation_config = segmentation_options[segmentation_type]
                                columns = {key: val for key, val in segmentation_config.items() if key != 'labels'}
                                segmentation_columns = list(columns.values())
                                labels = segmentation_config['labels']
                                
                                # Show original raw data without any processing
                                country_data_original = df[df['country'] == selected_country].copy()
                                columns_to_show = ['year'] + segmentation_columns
                                column_names = ['Year'] + labels
                                
                                display_data_original = country_data_original[columns_to_show].rename(columns=dict(zip(columns_to_show, column_names)))
                                st.dataframe(display_data_original)
                        else:
                            st.warning(f"No valid data available for {selected_country} with {segmentation_type} segmentation.")
                    else:
                        # Show instruction message
                        st.info("👆 Please select both a country and segmentation type to view the visualization.")
            else:
                st.warning(f"⚠️ No countries have data available for {st.session_state.selected_indicator}")
                st.info("Please select a different development outcome indicator.")
        else:
            st.warning("Could not load the dataset. Please check the file path.")
        
        # Back button
        if st.button("← Back to Main Dashboard", key="back_segmented_2"):
            st.session_state.show_segmented_trends = False
            st.rerun()
    
    elif st.session_state.show_country_comparison:
        # Show Country Comparison interface
        st.markdown("### 📈 Country Comparison Analysis")
        
        if df is not None:
            # Get countries that have data for the selected indicator
            countries = get_countries_with_data(df, st.session_state.selected_indicator)
            
            if countries:
                # Country selection dropdowns
                st.markdown("**Select Countries to Compare (maximum 3):**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    country1 = st.selectbox(
                        "Country 1:",
                        options=[""] + countries,
                        index=0,
                        help=f"Choose first country to compare {st.session_state.selected_indicator.lower()}",
                        key="comparison_country1"
                    )
                
                with col2:
                    country2 = st.selectbox(
                        "Country 2:",
                        options=[""] + countries,
                        index=0,
                        help=f"Choose second country to compare {st.session_state.selected_indicator.lower()}",
                        key="comparison_country2"
                    )
                
                with col3:
                    country3 = st.selectbox(
                        "Country 3 (Optional):",
                        options=[""] + countries,
                        index=0,
                        help=f"Choose third country to compare {st.session_state.selected_indicator.lower()}",
                        key="comparison_country3"
                    )
                
                # Filter out empty selections and limit to 3 countries
                selected_countries = [c for c in [country1, country2, country3] if c][:3]
                
                if len(selected_countries) >= 2:
                    # Show Information on Dataset
                    if st.session_state.selected_indicator in dataset_summaries:
                        with st.expander("📋 Information on Dataset", expanded=True):
                            st.text(dataset_summaries[st.session_state.selected_indicator])
                    
                    # Create and display the chart
                    chart = create_country_comparison_chart(df, selected_countries, st.session_state.selected_indicator)
                    if chart:
                        st.pyplot(chart)
                        
                        # Generate Insights button
                        st.markdown("---")
                        if st.button("🤖 Generate Insights", key="insights_comparison", type="secondary"):
                            with st.spinner("Generating insights..."):
                                # Prepare data for AI analysis
                                comparison_data = df[df['country'].isin(selected_countries)].sort_values(['country', 'year'])
                                indicator_info = OUTCOME_INDICATORS[st.session_state.selected_indicator]
                                total_column = indicator_info['total']
                                
                                # Filter data for the specific indicator
                                chart_data = comparison_data[['country', 'year', total_column]].dropna()
                                data_json = chart_data.to_json(orient='records')
                                
                                # Generate insights
                                insights = generate_insights(
                                    data_json=data_json,
                                    indicator_name=st.session_state.selected_indicator,
                                    countries_list=selected_countries,
                                    analysis_type="comparison"
                                )
                                
                                # Display insights
                                st.markdown("### 🤖 AI-Generated Insights")
                                st.info(insights)
                        
                        # Show data summary
                        st.markdown("### 📈 Comparison Summary")
                        
                        # Create metrics for each country
                        indicator_info = OUTCOME_INDICATORS[st.session_state.selected_indicator]
                        total_column = indicator_info['total']
                        unit = indicator_info['unit']
                        
                        cols = st.columns(len(selected_countries))
                        for i, country in enumerate(selected_countries):
                            with cols[i]:
                                country_data = df[df['country'] == country].copy()
                                # Ensure year column is numeric before sorting
                                country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
                                country_data = country_data.sort_values('year')
                                valid_data = country_data.dropna(subset=[total_column])
                                
                                if not valid_data.empty:
                                    latest_data_row = valid_data.iloc[-1]
                                    latest_year = latest_data_row['year']
                                    latest_value = latest_data_row[total_column]
                                    
                                    # Calculate trend
                                    if len(valid_data) > 1:
                                        first_value = valid_data[total_column].iloc[0]
                                        last_value = valid_data[total_column].iloc[-1]
                                        
                                        # Convert to numeric if needed
                                        try:
                                            first_value = float(first_value)
                                            last_value = float(last_value)
                                            trend = last_value - first_value
                                            year_start = int(valid_data['year'].min())
                                            year_end = int(valid_data['year'].max())
                                            st.metric(f"{country} ({year_start}-{year_end})", f"{trend:+.1f}{unit}")
                                        except (ValueError, TypeError):
                                            st.metric(f"{country}", "N/A")
                                    else:
                                        st.metric(f"{country}", "Single data point")
                                else:
                                    st.metric(f"{country}", "No valid data")
                        
                        # Show raw data
                        with st.expander("View Raw Data"):
                            comparison_data = df[df['country'].isin(selected_countries)].sort_values(['country', 'year'])
                            display_data = comparison_data[['country', 'year', total_column]].copy()
                            
                            # Filter out years where ALL countries have no data
                            years_with_data = display_data.dropna(subset=[total_column])['year'].unique()
                            if len(years_with_data) > 0:
                                filtered_data = display_data[display_data['year'].isin(years_with_data)]
                                filtered_data = filtered_data.rename(columns={
                                    total_column: f'{indicator_info["display_name"]} ({unit})'
                                })
                                st.dataframe(filtered_data)
                            else:
                                st.info("No valid data available for comparison.")
                    else:
                        st.warning("No valid data available for the selected countries.")
                else:
                    st.info("👆 Please select at least 2 countries to view the comparison (maximum 3).")
            else:
                st.warning(f"⚠️ No countries have data available for {st.session_state.selected_indicator}")
                st.info("Please select a different development outcome indicator.")
        else:
            st.warning("Could not load the dataset. Please check the file path.")
        
        # Back button
        if st.button("← Back to Main Dashboard", key="back_comparison"):
            st.session_state.show_country_comparison = False
            st.rerun()
    
    elif st.session_state.show_global_benchmarking:
        # Show Global Benchmarking interface
        st.markdown("### ⚖️ Country Benchmarking (Regional and Global) Analysis")
        
        if df is not None:
            # Get countries that have data for the selected indicator
            countries = get_countries_with_data(df, st.session_state.selected_indicator)
            
            if countries:
                # Country selection dropdown
                selected_country = st.selectbox(
                    "Select a Country:",
                    options=countries,
                    index=0,
                    help=f"Choose a country to compare against the global average for {st.session_state.selected_indicator.lower()}",
                    key="benchmarking_country"
                )
                
                if selected_country:
                    # Show Information on Dataset
                    if st.session_state.selected_indicator in dataset_summaries:
                        with st.expander("📋 Information on Dataset", expanded=True):
                            st.text(dataset_summaries[st.session_state.selected_indicator])
                    
                    # Create and display the line chart
                    chart = create_global_benchmarking_chart(df, selected_country, st.session_state.selected_indicator)
                    if chart:
                        st.pyplot(chart)
                        
                        # Generate Insights button
                        st.markdown("---")
                        if st.button("🤖 Generate Insights", key="insights_benchmarking", type="secondary"):
                            with st.spinner("Generating insights..."):
                                # Prepare data for AI analysis
                                country_data = df[df['country'] == selected_country].copy()
                                # Ensure year column is numeric before sorting
                                country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
                                country_data = country_data.sort_values('year')
                                global_averages = calculate_global_averages(df, st.session_state.selected_indicator)
                                indicator_info = OUTCOME_INDICATORS[st.session_state.selected_indicator]
                                total_column = indicator_info['total']
                                
                                # Prepare country data
                                country_chart_data = country_data[['year', total_column]].dropna()
                                
                                # Prepare global averages data
                                if global_averages is not None and not global_averages.empty:
                                    global_chart_data = global_averages[['year', 'global_average']].rename(columns={'global_average': total_column})
                                    
                                    # Combine data for analysis
                                    combined_data = {
                                        'country_data': country_chart_data.to_dict('records'),
                                        'global_averages': global_chart_data.to_dict('records')
                                    }
                                    data_json = json.dumps(combined_data)
                                else:
                                    data_json = country_chart_data.to_json(orient='records')
                                
                                # Generate insights
                                insights = generate_insights(
                                    data_json=data_json,
                                    indicator_name=st.session_state.selected_indicator,
                                    country_name=selected_country,
                                    analysis_type="benchmarking"
                                )
                                
                                # Display insights
                                st.markdown("### 🤖 AI-Generated Insights")
                                st.info(insights)
                    
                    # Create and display the bar chart for last 3 years
                    st.markdown("---")
                    st.markdown("### 📊 Recent Performance Comparison (Last 3 Years)")
                    bar_chart = create_benchmarking_bar_chart(df, selected_country, st.session_state.selected_indicator)
                    if bar_chart:
                        st.pyplot(bar_chart)
                        
                        # Generate Insights button for bar chart
                        st.markdown("---")
                        if st.button("🤖 Generate Insights", key="insights_bar_chart", type="secondary"):
                            with st.spinner("Generating insights..."):
                                # Prepare data for AI analysis (last 3 years)
                                country_data = df[df['country'] == selected_country].copy()
                                # Ensure year column is numeric before sorting
                                country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
                                country_data[total_column] = pd.to_numeric(country_data[total_column], errors='coerce')
                                country_data = country_data.sort_values('year')
                                global_averages = calculate_global_averages(df, st.session_state.selected_indicator)
                                indicator_info = OUTCOME_INDICATORS[st.session_state.selected_indicator]
                                total_column = indicator_info['total']
                                
                                # Get last 3 years data
                                last_3_years_data = country_data.dropna(subset=[total_column]).tail(3)
                                
                                if not last_3_years_data.empty and global_averages is not None:
                                    # Filter global averages for the same years
                                    last_3_years = last_3_years_data['year'].unique()
                                    global_averages['year'] = pd.to_numeric(global_averages['year'], errors='coerce')
                                    global_averages_filtered = global_averages[global_averages['year'].isin(last_3_years)]
                                    
                                    # Prepare combined data
                                    country_chart_data = last_3_years_data[['year', total_column]]
                                    global_chart_data = global_averages_filtered[['year', 'global_average']].rename(columns={'global_average': total_column})
                                    
                                    combined_data = {
                                        'country_data': country_chart_data.to_dict('records'),
                                        'global_averages': global_chart_data.to_dict('records'),
                                        'analysis_period': 'last_3_years'
                                    }
                                    data_json = json.dumps(combined_data)
                                else:
                                    data_json = last_3_years_data[['year', total_column]].to_json(orient='records')
                                
                                # Generate insights
                                insights = generate_insights(
                                    data_json=data_json,
                                    indicator_name=st.session_state.selected_indicator,
                                    country_name=selected_country,
                                    analysis_type="benchmarking"
                                )
                                
                                # Display insights
                                st.markdown("### 🤖 AI-Generated Insights")
                                st.info(insights)
                    else:
                        st.info("Bar chart not available - insufficient data for the last 3 years.")
                    
                    # Create and display the regional comparison chart
                    st.markdown("---")
                    st.markdown("### 🌍 Regional Comparison (Latest Year)")
                    regional_chart = create_regional_comparison_chart(df, selected_country, st.session_state.selected_indicator)
                    if regional_chart:
                        st.pyplot(regional_chart)
                        
                        # Generate Insights button for regional comparison
                        st.markdown("---")
                        if st.button("🤖 Generate Insights", key="insights_regional_chart", type="secondary"):
                            with st.spinner("Generating insights..."):
                                # Prepare data for AI analysis (regional comparison)
                                country_data = df[df['country'] == selected_country].copy()
                                # Ensure year column is numeric before sorting
                                country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
                                country_data[total_column] = pd.to_numeric(country_data[total_column], errors='coerce')
                                country_data = country_data.sort_values('year')
                                indicator_info = OUTCOME_INDICATORS[st.session_state.selected_indicator]
                                total_column = indicator_info['total']
                                
                                # Get the latest year with data for the selected country
                                latest_year = country_data.dropna(subset=[total_column])['year'].max()
                                
                                # Get regional data for the latest year
                                region = country_data['region'].iloc[0]
                                regional_data = df[df['region'] == region].copy()
                                regional_data['year'] = pd.to_numeric(regional_data['year'], errors='coerce')
                                regional_data[total_column] = pd.to_numeric(regional_data[total_column], errors='coerce')
                                
                                latest_year_data = regional_data[
                                    (regional_data['year'] == latest_year) & 
                                    (regional_data[total_column].notna())
                                ].sort_values(total_column, ascending=False)
                                
                                if not latest_year_data.empty:
                                    # Prepare regional comparison data
                                    regional_chart_data = latest_year_data[['country', total_column]].copy()
                                    regional_chart_data['is_selected_country'] = regional_chart_data['country'] == selected_country
                                    
                                    combined_data = {
                                        'regional_data': regional_chart_data.to_dict('records'),
                                        'selected_country': selected_country,
                                        'region': region,
                                        'latest_year': int(latest_year),
                                        'analysis_type': 'regional_comparison'
                                    }
                                    data_json = json.dumps(combined_data)
                                else:
                                    data_json = f'{{"error": "No regional data available for {latest_year}"}}'
                                
                                # Generate insights
                                insights = generate_insights(
                                    data_json=data_json,
                                    indicator_name=st.session_state.selected_indicator,
                                    country_name=selected_country,
                                    analysis_type="benchmarking"
                                )
                                
                                # Display insights
                                st.markdown("### 🤖 AI-Generated Insights")
                                st.info(insights)
                    else:
                        st.info("Regional comparison chart not available - insufficient regional data.")
                        
                        # Show data summary
                        st.markdown("### 📈 Benchmarking Summary")
                        
                        # Get country data and global averages
                        country_data = df[df['country'] == selected_country].copy()
                        # Ensure year column is numeric before sorting
                        country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
                        country_data = country_data.sort_values('year')
                        global_averages = calculate_global_averages(df, st.session_state.selected_indicator)
                        
                        indicator_info = OUTCOME_INDICATORS[st.session_state.selected_indicator]
                        total_column = indicator_info['total']
                        unit = indicator_info['unit']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Years of data for selected country
                            valid_country_data = country_data.dropna(subset=[total_column])
                            st.metric("Country Data Points", len(valid_country_data))
                        
                        with col2:
                            # Latest comparison
                            if not valid_country_data.empty and global_averages is not None:
                                latest_country_data = valid_country_data.iloc[-1]
                                latest_country_year = latest_country_data['year']
                                latest_country_value = latest_country_data[total_column]
                                
                                # Convert to numeric if needed
                                try:
                                    latest_country_value = float(latest_country_value)
                                except (ValueError, TypeError):
                                    latest_country_value = None
                                
                                # Find closest global average year
                                # Ensure consistent data types for comparison
                                latest_country_year_numeric = pd.to_numeric(latest_country_year, errors='coerce')
                                global_averages['year'] = pd.to_numeric(global_averages['year'], errors='coerce')
                                global_year_data = global_averages[global_averages['year'] == latest_country_year_numeric]
                                if not global_year_data.empty and latest_country_value is not None:
                                    latest_global_value = float(global_year_data['global_average'].iloc[0])
                                    difference = latest_country_value - latest_global_value
                                    st.metric(f"vs Global Avg ({latest_country_year})", f"{difference:+.1f}{unit}")
                                else:
                                    st.metric("vs Global Avg", "No comparable data")
                            else:
                                st.metric("vs Global Avg", "No valid data")
                        
                        with col3:
                            # Overall performance
                            if not valid_country_data.empty and global_averages is not None:
                                # Calculate average difference across all years
                                country_years = valid_country_data['year'].unique()
                                differences = []
                                
                                for year in country_years:
                                    country_value = valid_country_data[valid_country_data['year'] == year][total_column].iloc[0]
                                    # Ensure consistent data types for comparison
                                    year_numeric = pd.to_numeric(year, errors='coerce')
                                    global_year_data = global_averages[global_averages['year'] == year_numeric]
                                    if not global_year_data.empty:
                                        try:
                                            country_value = float(country_value)
                                            global_value = float(global_year_data['global_average'].iloc[0])
                                            differences.append(country_value - global_value)
                                        except (ValueError, TypeError):
                                            continue  # Skip non-numeric values
                                
                                if differences:
                                    avg_difference = sum(differences) / len(differences)
                                    st.metric("Average Performance", f"{avg_difference:+.1f}{unit}")
                                else:
                                    st.metric("Average Performance", "N/A")
                            else:
                                st.metric("Average Performance", "N/A")
                        
                        # Show raw data comparison
                        with st.expander("View Raw Data Comparison"):
                            if global_averages is not None:
                                # Merge country and global data
                                comparison_data = valid_country_data[['year', total_column]].copy()
                                
                                # Ensure both dataframes have consistent data types for the year column
                                comparison_data['year'] = pd.to_numeric(comparison_data['year'], errors='coerce')
                                global_averages_copy = global_averages[['year', 'global_average', 'country_count']].copy()
                                global_averages_copy['year'] = pd.to_numeric(global_averages_copy['year'], errors='coerce')
                                
                                comparison_data = comparison_data.merge(
                                    global_averages_copy, 
                                    on='year', 
                                    how='left'
                                )
                                
                                # Calculate difference (ensure numeric conversion)
                                comparison_data[total_column] = pd.to_numeric(comparison_data[total_column], errors='coerce')
                                comparison_data['global_average'] = pd.to_numeric(comparison_data['global_average'], errors='coerce')
                                comparison_data['difference'] = comparison_data[total_column] - comparison_data['global_average']
                                
                                # Rename columns for display
                                display_data = comparison_data.rename(columns={
                                    total_column: f'{selected_country} ({unit})',
                                    'global_average': f'Global Average ({unit})',
                                    'country_count': 'Countries in Global Avg',
                                    'difference': f'Difference ({unit})'
                                })
                                
                                st.dataframe(display_data)
                            else:
                                st.info("Could not calculate global averages for comparison.")
            else:
                st.warning(f"⚠️ No countries have data available for {st.session_state.selected_indicator}")
                st.info("Please select a different development outcome indicator.")
        else:
            st.warning("Could not load the dataset. Please check the file path.")
        
        # Back button
        if st.button("← Back to Main Dashboard", key="back_benchmarking"):
            st.session_state.show_global_benchmarking = False
            st.rerun()
    
    elif st.session_state.show_peer_benchmarking:
        # Show Peer Country Benchmarking interface
        st.markdown("### 👥 Peer Country Benchmarking Analysis")
        
        if df is not None:
            # Get countries that have peer country data
            countries_with_peers = df[df['Country 2'].notna()]['country'].unique()
            
            if len(countries_with_peers) > 0:
                # Country selection dropdown
                selected_country = st.selectbox(
                    "Select a Country:",
                    options=sorted(countries_with_peers),
                    index=0,
                    help=f"Choose a country to compare against its peer countries for {st.session_state.selected_indicator.lower()}",
                    key="peer_benchmarking_country"
                )
                
                if selected_country:
                    # Show Information on Dataset
                    if st.session_state.selected_indicator in dataset_summaries:
                        with st.expander("📋 Information on Dataset", expanded=True):
                            st.text(dataset_summaries[st.session_state.selected_indicator])
                    
                    # Create and display the charts
                    chart_result = create_peer_country_comparison_chart(df, selected_country, st.session_state.selected_indicator)
                    if chart_result is not None:
                        chart, explanation, countries_with_data = chart_result
                        if chart:
                            st.pyplot(chart)
                            
                            # Generate Insights button for line chart (time trends) - directly below line chart
                            st.markdown("---")
                            st.markdown("#### 🤖 Generate Insights for Time Trends")
                            if st.button("🤖 Generate Insights", key="insights_line_chart", type="secondary"):
                                with st.spinner("Generating insights..."):
                                    # Prepare data for AI analysis using countries_with_data
                                    comparison_data = df[df['country'].isin(countries_with_data)].sort_values(['country', 'year'])
                                    indicator_info = OUTCOME_INDICATORS[st.session_state.selected_indicator]
                                    total_column = indicator_info['total']
                                    
                                    # Filter data for the specific indicator
                                    chart_data = comparison_data[['country', 'year', total_column]].dropna()
                                    data_json = chart_data.to_json(orient='records')
                                    
                                    # Generate insights
                                    insights = generate_insights(
                                        data_json=data_json,
                                        indicator_name=st.session_state.selected_indicator,
                                        country_name=selected_country,
                                        analysis_type="peer_comparison"
                                    )
                                    
                                    # Display insights
                                    st.markdown("### 🤖 AI-Generated Insights")
                                    st.info(insights)
                            
                            # Create and display the bar chart
                            st.markdown("---")
                            st.markdown("### 📊 Latest Year Comparison")
                            bar_chart = create_peer_country_bar_chart(df, selected_country, st.session_state.selected_indicator, countries_with_data)
                            if bar_chart:
                                st.pyplot(bar_chart)
                                
                                # Generate Insights button for bar chart - directly below bar chart
                                st.markdown("---")
                                st.markdown("#### 🤖 Generate Insights for Latest Year Comparison")
                                if st.button("🤖 Generate Insights", key="insights_bar_chart", type="secondary"):
                                    with st.spinner("Generating insights..."):
                                        # Prepare data for AI analysis (latest year comparison)
                                        comparison_data = df[df['country'].isin(countries_with_data)].copy()
                                        indicator_info = OUTCOME_INDICATORS[st.session_state.selected_indicator]
                                        total_column = indicator_info['total']
                                        
                                        # Ensure year and indicator columns are numeric
                                        comparison_data['year'] = pd.to_numeric(comparison_data['year'], errors='coerce')
                                        comparison_data[total_column] = pd.to_numeric(comparison_data[total_column], errors='coerce')
                                        
                                        # Remove rows with NaN values for the indicator
                                        comparison_data = comparison_data.dropna(subset=[total_column])
                                        
                                        # Find the latest year where ALL countries have data
                                        all_years = sorted(comparison_data['year'].unique(), reverse=True)
                                        
                                        latest_common_year = None
                                        for year in all_years:
                                            year_data = comparison_data[comparison_data['year'] == year]
                                            countries_in_year = year_data['country'].unique()
                                            
                                            if all(country in countries_in_year for country in countries_with_data):
                                                latest_common_year = year
                                                break
                                        
                                        if latest_common_year is not None:
                                            # Get data for the latest common year
                                            latest_year_data = comparison_data[comparison_data['year'] == latest_common_year].copy()
                                            
                                            if not latest_year_data.empty:
                                                # Prepare bar chart data for analysis
                                                bar_chart_data = latest_year_data[['country', total_column]].copy()
                                                bar_chart_data['year'] = latest_common_year
                                                
                                                combined_data = {
                                                    'bar_chart_data': bar_chart_data.to_dict('records'),
                                                    'selected_country': selected_country,
                                                    'latest_year': int(latest_common_year),
                                                    'analysis_type': 'latest_year_comparison'
                                                }
                                                data_json = json.dumps(combined_data)
                                            else:
                                                data_json = f'{{"error": "No data available for {latest_common_year}"}}'
                                        else:
                                            data_json = '{"error": "No latest year data available"}'
                                        
                                        # Generate insights
                                        insights = generate_insights(
                                            data_json=data_json,
                                            indicator_name=st.session_state.selected_indicator,
                                            country_name=selected_country,
                                            analysis_type="peer_comparison"
                                        )
                                        
                                        # Display insights
                                        st.markdown("### 🤖 AI-Generated Insights")
                                        st.info(insights)
                            else:
                                st.info("Bar chart not available - insufficient data for latest year comparison.")
                            
                            # Show explanation
                            if explanation:
                                st.markdown("---")
                                st.markdown("### 📝 Why These Countries Were Selected")
                                st.info(explanation)
                            
                            # Show data summary
                            st.markdown("### 📈 Peer Comparison Summary")
                            
                            # Create metrics for each country using countries_with_data
                            indicator_info = OUTCOME_INDICATORS[st.session_state.selected_indicator]
                            total_column = indicator_info['total']
                            unit = indicator_info['unit']
                            
                            cols = st.columns(len(countries_with_data))
                            for i, country in enumerate(countries_with_data):
                                with cols[i]:
                                    country_data = df[df['country'] == country].copy()
                                    # Ensure year column is numeric before sorting
                                    country_data['year'] = pd.to_numeric(country_data['year'], errors='coerce')
                                    country_data = country_data.sort_values('year')
                                    valid_data = country_data.dropna(subset=[total_column])
                                    
                                    if not valid_data.empty:
                                        latest_data_row = valid_data.iloc[-1]
                                        latest_year = latest_data_row['year']
                                        latest_value = latest_data_row[total_column]
                                        
                                        # Calculate trend
                                        if len(valid_data) > 1:
                                            first_value = valid_data[total_column].iloc[0]
                                            last_value = valid_data[total_column].iloc[-1]
                                            
                                            # Convert to numeric if needed
                                            try:
                                                first_value = float(first_value)
                                                last_value = float(last_value)
                                                trend = last_value - first_value
                                                year_start = int(valid_data['year'].min())
                                                year_end = int(valid_data['year'].max())
                                                st.metric(f"{country} ({year_start}-{year_end})", f"{trend:+.1f}{unit}")
                                            except (ValueError, TypeError):
                                                st.metric(f"{country}", "N/A")
                                        else:
                                            st.metric(f"{country}", "Single data point")
                                    else:
                                        st.metric(f"{country}", "No valid data")
                            
                            # Show raw data
                            with st.expander("View Raw Data"):
                                comparison_data = df[df['country'].isin(countries_with_data)].sort_values(['country', 'year'])
                                display_data = comparison_data[['country', 'year', total_column]].copy()
                                
                                # Filter out years where ALL countries have no data
                                years_with_data = display_data.dropna(subset=[total_column])['year'].unique()
                                if len(years_with_data) > 0:
                                    filtered_data = display_data[display_data['year'].isin(years_with_data)]
                                    filtered_data = filtered_data.rename(columns={
                                        total_column: f'{indicator_info["display_name"]} ({unit})'
                                    })
                                    st.dataframe(filtered_data)
                                else:
                                    st.info("No valid data available for comparison.")
                        else:
                            st.warning("No valid data available for peer country comparison.")
                    else:
                        st.warning("No peer countries available for the selected country.")
            else:
                st.warning("⚠️ No countries have peer country data available")
                st.info("Peer country comparisons are not available for any countries in the dataset.")
        else:
            st.warning("Could not load the dataset. Please check the file path.")
        
        # Back button
        if st.button("← Back to Main Dashboard", key="back_peer_benchmarking"):
            st.session_state.show_peer_benchmarking = False
            st.rerun()
    
    else:
        # Show main dashboard with squares
        # Create the 4 visualization squares
        st.markdown('<div class="squares-container">', unsafe_allow_html=True)
        
        # Square 1: Country Specific Time-Trend
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Country Specific Time-Trend", key="btn1", use_container_width=True):
                st.session_state.show_time_trends = True
                st.rerun()
        
        # Square 2: Country Specific Time-Trends Segmented
        with col2:
            if st.button("🔍 Country Specific Time-Trends Segmented", key="btn2", use_container_width=True):
                st.session_state.show_segmented_trends = True
                st.rerun()
        
        # Square 3: Country Comparisons
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("📈 Country Comparisons", key="btn3", use_container_width=True):
                st.session_state.show_country_comparison = True
                st.rerun()
        
        # Square 4: Country Benchmarking (Regional and Global)
        with col4:
            if st.button("⚖️ Country Benchmarking (Regional and Global)", key="btn4", use_container_width=True):
                st.session_state.show_global_benchmarking = True
                st.rerun()
        
        # Square 5: Peer Country Benchmarking
        col5, col6 = st.columns(2)
        
        with col5:
            if st.button("👥 Country Benchmarking (Peer Countries)", key="btn5", use_container_width=True):
                st.session_state.show_peer_benchmarking = True
                st.rerun()
        
        # Empty column for layout
        with col6:
            st.empty()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer information
    st.markdown("---")

if __name__ == "__main__":
    main()
