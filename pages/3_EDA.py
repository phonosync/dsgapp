import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import utils

@st.cache_data
def load_data(uploaded_file):
    return utils.inp_file_2_csv(inp_file=uploaded_file, index_column=False,
                                  header_row=True)

def detect_variable_type(df):
    variable_types = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            variable_types[column] = "Numerical"
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            variable_types[column] = "DateTime"
        else:
            variable_types[column] = "Categorical"
    return variable_types

st.title("Explorative Datenanalyse")
st.write("Für tabellarische Daten aus einer csv-Datei")

# File uploader
uploaded_file = st.file_uploader("Wähle eine csv-Datei", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = load_data(uploaded_file)
    
    # Calculate statistics
    num_rows, num_columns = df.shape
    num_missing_cells = df.isna().sum().sum()
    total_cells = num_rows * num_columns
    perc_missing_cells = (num_missing_cells / total_cells) * 100
    num_duplicate_rows = df.duplicated().sum()
    perc_duplicate_rows = (num_duplicate_rows / num_rows) * 100
    
    # Create a summary table
    summary_table = pd.DataFrame({
        "Metrik": [
            "Anzahl Spalten",
            "Anzahl Zeilen",
            "Anzahl leerer Zellen",
            "Anteil (%) leerer Tellen",
            "Anzahl duplizierter Zeilen",
            "Anteil (%) duplizierter Zeilen"
        ],
        "Wert": [
            str(num_columns),
            str(num_rows),
            str(num_missing_cells),
            f"{perc_missing_cells:.2f}%",
            str(num_duplicate_rows),
            f"{perc_duplicate_rows:.2f}%"
        ]
    })

    # Detect variable types
    variable_types = detect_variable_type(df)

    # Create a DataFrame with the number of columns per datatype
    datatype_summary = pd.DataFrame({
        "Datentyp": ["Numerisch", "Kategorisch", "Datum"],
        "Anzahl Spalten": [
            sum(1 for v in variable_types.values() if v == "Numerical"),
            sum(1 for v in variable_types.values() if v == "Categorical"),
            sum(1 for v in variable_types.values() if v == "DateTime")
        ]
    }).sort_values(by="Anzahl Spalten", ascending=False)
    
    st.header('Übersicht')
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(summary_table, hide_index=True)
    with col2:
        st.dataframe(datatype_summary, hide_index=True)
    
    # Display sample of the first 10 rows
    st.write("Auszug aus Datensatz:")
    st.dataframe(df.head(10), hide_index=True)

    st.header('Univariate Analyse')

    # Select a column to analyze
    selected_column = st.selectbox("Wähle die zu analysierende Spalte", df.columns)
    
    if selected_column:
        # Get the detected variable type for the selected column
        detected_type = variable_types[selected_column]
        
        # Select variable type for the selected column, with the detected type pre-selected
        variable_type = st.selectbox(f"Wähle Datentyp für {selected_column}", ["Numerisch", "Kategorisch", "Datum"], index=["Numerical", "Categorical", "Timeseries"].index(detected_type))
        
        if variable_type == "Numerisch":
            # Overview
            distinct_values = df[selected_column].nunique()
            missing_values = df[selected_column].isna().sum()
            zeros = (df[selected_column] == 0).sum()
            negative_values = (df[selected_column] < 0).sum()

            total_values = len(df[selected_column])
            perc_distinct_values = np.round((distinct_values / total_values) * 100, 2)
            perc_missing_values = np.round((missing_values / total_values) * 100, 2)
            perc_zeros = np.round((zeros / total_values) * 100, 2)
            perc_negative_values = np.round((negative_values / total_values) * 100, 2)

            # Create DataFrame for additional metrics
            df3 = pd.DataFrame({
                "Metrik": ["Eindeutige Werte", "Fehlende Werte oder NaN", "Nullwerte", "Negative Werte"],
                "Anzahl": [distinct_values, missing_values, zeros, negative_values],
                "Anteil (%)": [perc_distinct_values, perc_missing_values, perc_zeros, perc_negative_values]
            })

            # Display the additional metrics dataframe
            st.subheader('Übersicht Numerische Variable')
            st.dataframe(df3, hide_index=True)

            # Calculate descriptive statistics
            min_val = df[selected_column].min()
            percentile_5 = df[selected_column].quantile(0.05)
            q1 = df[selected_column].quantile(0.25)
            median = df[selected_column].median()
            q3 = df[selected_column].quantile(0.75)
            percentile_95 = df[selected_column].quantile(0.95)
            max_val = df[selected_column].max()
            range_val = max_val - min_val
            iqr = q3 - q1

            mean = np.round(df[selected_column].mean(), 2)
            std_dev = np.round(df[selected_column].std(), 2)
            variance = np.round(df[selected_column].var(), 2)
            cv = np.round(std_dev / mean, 2)
            kurtosis = np.round(df[selected_column].kurtosis(), 2)
            skewness = np.round(df[selected_column].skew(), 2)

            # Create DataFrame 1
            df1 = pd.DataFrame({
                "Metrik": ["Minimum", "5. Perzentil", "Q1", "Median", "Q3", "95. Perzentil", "Maximum", "Spannweite", "Interquartile Range (IQR)"],
                "Wert": [min_val, percentile_5, q1, median, q3, percentile_95, max_val, range_val, iqr]
            })

            # Create DataFrame 2
            df2 = pd.DataFrame({
                "Metrik": ["Mean", "Standardabweichung", "Varianz", "Coefficient of Variation (CV)", "Kurtosis", "Skewness"],
                "Wert": [mean, std_dev, variance, cv, kurtosis, skewness]
            })

            st.subheader("Descriptive Metriken")
            # Display the dataframes side by side
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df1, hide_index=True)
            with col2:
                st.dataframe(df2, hide_index=True)
            
            # Histogram
            st.subheader('Histogramm')
            # Drop NaN values for histogram calculation
            cleaned_data = df[selected_column].dropna()
            # Calculate default number of bins using numpy
            default_bins = np.histogram_bin_edges(cleaned_data, bins='auto').size - 1
            bins = st.slider('Wähle die Anzahl Bins', min_value=1, max_value=100, value=default_bins)

            # Display histogram
            fig, ax = plt.subplots()
            cleaned_data.hist(ax=ax, bins=bins)
            ax.set_title(f'Histogramm')
            ax.set_ylabel('Häufigkeit')
            ax.set_xlabel(selected_column)
            ax.grid(False)  # Remove grid
            ax.spines['top'].set_visible(False)  # Remove top spine
            ax.spines['right'].set_visible(False)  # Remove right spine
            st.pyplot(fig)

            # Display box plot
            # st.subheader("Box plot")
            fig, ax = plt.subplots()
            
            # Box plot
            sns.boxplot(x=np.ones(len(cleaned_data)), y=cleaned_data,
                        color='lightgrey', linewidth=1.5,
                        medianprops={'color': 'black', 'linewidth': 2},
                        flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 5,
                                    'markeredgecolor': 'black'},
                        ax=ax)
            
            ax.set_xlim(-0.5, 5)
            ax.set_xticks([])
            # ax.set_yticks([])
            # ax.tick_params(axis='both', bottom=False, left=False)
            for loc in ['top', 'right', 'bottom']:
                ax.spines[loc].set_visible(False)
            # ax.set_ylabel(selected_column, rotation=0, labelpad=20)
            # ax.set_ylabel('')
            # Add a custom y-label in the top left corner
            #ax.annotate(selected_column, xy=(0, 1), xytext=(-30, 15),
            #            xycoords='axes fraction', textcoords='offset points',
            #            ha='left', va='center', fontsize=12, rotation=0)
            # ax.set_title('Boxplot')

            ax.set_ylabel('')
            # Add a custom y-label above the y-axis tick labels
            y_label = ax.annotate(selected_column, xy=(0, 1), xytext=(-10, 10),
                                xycoords='axes fraction', textcoords='offset points',
                                ha='right', va='bottom', fontsize=12, rotation=0)

            # Calculate the space needed for the y-label
            renderer = fig.canvas.get_renderer()
            bbox = y_label.get_window_extent(renderer=renderer)
            text_width = bbox.width / fig.dpi  # Convert from pixels to inches

            # Ensure the left margin does not exceed a reasonable percentage of the figure width
            max_left_margin = 0.3 * fig.get_size_inches()[0]  # 30% of the figure width
            left_margin = min(text_width + 0.1, max_left_margin)  # Add some padding

            # Adjust the plot margins
            fig.subplots_adjust(left=left_margin / fig.get_size_inches()[0])

            st.pyplot(fig)
        
        elif variable_type == "Kategorisch":

            num_distinct_classes = df[selected_column].nunique()
            
            st.write(f'Anzahl Klassen: {num_distinct_classes}')

            # Construct a DataFrame with absolute and relative frequency per class
            frequency_df = pd.DataFrame({
                "Klasse": df[selected_column].value_counts().index,
                "Absolute Häufigkeit": df[selected_column].value_counts().values,
                "Relative Häufigkeit (%)": (df[selected_column].value_counts(normalize=True) * 100).values.round(2)
            }).sort_values(by="Absolute Häufigkeit", ascending=False)
            
            st.dataframe(frequency_df, hide_index=True)

            # Input field for the number of most frequent classes to display
            num_classes = st.number_input("Anzahl der häufigsten Klassen, die explizit dargestellt werden sollen:", min_value=1, max_value=len(frequency_df), value=5)
            
            # Sort the DataFrame by "Absolute Häufigkeit" in descending order
            sorted_df = frequency_df.sort_values(by="Absolute Häufigkeit", ascending=False)
            
            # Select the top N classes
            top_classes_df = sorted_df.head(num_classes)
            other_classes_df = sorted_df.iloc[num_classes:]
            
            # Add "Sonstige" class
            if not other_classes_df.empty:
                other_row = pd.DataFrame({
                    "Klasse": ["Sonstige"],
                    "Absolute Häufigkeit": [other_classes_df["Absolute Häufigkeit"].sum()],
                    "Relative Häufigkeit (%)": [other_classes_df["Relative Häufigkeit (%)"].sum()]
                })
                top_classes_df = pd.concat([top_classes_df, other_row], ignore_index=True)
            
            # Frequency distribution
            st.write(f"Anteil und Häufigkeitsverteilung der {num_classes} häufigsten Klassen:")

            # Sort the DataFrame by "Absolute Häufigkeit" in descending order
            sorted_df = frequency_df.sort_values(by="Absolute Häufigkeit", ascending=False)

            # Select the top N classes
            top_classes_df = sorted_df.head(num_classes)
            other_classes_df = sorted_df.iloc[num_classes:]

            # Add "Sonstige" class
            if not other_classes_df.empty:
                other_row = pd.DataFrame({
                    "Klasse": ["Sonstige"],
                    "Absolute Häufigkeit": [other_classes_df["Absolute Häufigkeit"].sum()],
                    "Relative Häufigkeit (%)": [other_classes_df["Relative Häufigkeit (%)"].sum()]
                })
                top_classes_df = pd.concat([top_classes_df, other_row], ignore_index=True)

            # Define a color palette
            colors = plt.cm.tab20.colors  # You can choose any colormap you prefer
            
            col21, col22 = st.columns(2)

            # Define the figure size
            fig_width, fig_height = 6, 6

            # Pie chart
            with col21:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                top_classes_df.set_index("Klasse")["Absolute Häufigkeit"].plot.pie(ax=ax, autopct='%1.2f%%', startangle=90, colors=colors[:len(top_classes_df)])
                ax.set_ylabel('')
                ax.set_aspect('equal')  # Ensure the pie chart is a circle

                st.pyplot(fig)
            # Horizontal bar chart (excluding "Sonstige")
            with col22:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                bar_chart_df = top_classes_df[top_classes_df["Klasse"] != "Sonstige"]
                bar_chart_df = bar_chart_df.sort_values(by="Absolute Häufigkeit", ascending=True)  # Sort in ascending order for top-to-bottom display
                bar_chart_df.set_index("Klasse")["Absolute Häufigkeit"].plot.barh(ax=ax, color=colors[:len(bar_chart_df)][::-1])
                # Remove y-tick marks but keep labels
                ax.tick_params(axis='y', length=0)

                ax.set_xlabel('Häufigkeit')
                ax.set_ylabel('Klasse', rotation=0)
                ax.yaxis.set_label_coords(-0.1, 1.02)  # Move the label to the top

                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)


                st.pyplot(fig)
        
        elif variable_type == "Datum":
            # Select a column to be interpreted as datetime
            datetime_column = st.selectbox("Wähle eine als Datum zu interpretierende Spalte", df.columns)
            
            if datetime_column:
                df[datetime_column] = pd.to_datetime(df[datetime_column])
                df.set_index(datetime_column, inplace=True)
                
                # Display timeseries plot
                # st.write(f"Timeseries plot for {selected_column}:")
                fig, ax = plt.subplots()
                ax.set_title(f"Zeitreihe für {selected_column}:")
                df[selected_column].plot(ax=ax)
                st.pyplot(fig)

    st.header('Wechselwirkungen')

    col1, col2 = st.columns(2)
    with col1:
        # Select a column to analyze
        sel_col_1 = st.selectbox("Select a column to analyze", df.columns, key='fortytwo')
        if sel_col_1:
            # Get the detected variable type for the selected column
            det_type_1 = variable_types[sel_col_1]
        
            # Select variable type for the selected column, with the detected type pre-selected
            var_type_1 = st.selectbox(f"Adjust type for {sel_col_1}", ["Numerical", "Categorical", "DateTime"],
                                        index=["Numerical", "Categorical", "DateTime"].index(det_type_1),
                                        key='fortythree')
    
    with col2:
        # Select a column to analyze
        sel_col_2 = st.selectbox("Select a column to analyze", df.columns, key='fortyfour')
        if sel_col_2:
            # Get the detected variable type for the selected column
            det_type_2 = variable_types[sel_col_2]
        
            # Select variable type for the selected column, with the detected type pre-selected
            var_type_2 = st.selectbox(f"Adjust type for {sel_col_2}", ["Numerical", "Categorical", "DateTime"],
                                        index=["Numerical", "Categorical", "DateTime"].index(det_type_2),
                                        key='fortyfive')
    
    # Drop rows where either sel_col_1 or sel_col_2 is NaN
    cleaned_df = df.dropna(subset=[sel_col_1, sel_col_2])

    if var_type_1=='Numerical' and var_type_2=='Numerical':

        # Calculate Pearson and Spearman correlation coefficients
        pearson_corr, _ = pearsonr(cleaned_df[sel_col_1], cleaned_df[sel_col_2])
        spearman_corr, _ = spearmanr(cleaned_df[sel_col_1], cleaned_df[sel_col_2])

        st.write('Correlation coefficients:')
        st.dataframe(pd.DataFrame(data=[[f'{pearson_corr:.2f}', f'{spearman_corr:.2f}']],
                    columns=['Pearson:', 'Spearman:']), hide_index=True) 

        # Make a scatterplot
        # st.write(f"Scatterplot:")
        fig, ax = plt.subplots()
        ax.scatter(cleaned_df[sel_col_1], cleaned_df[sel_col_2])
        ax.set_xlabel(sel_col_1)

        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)

        ax.set_ylabel('')
        # Add a custom y-label above the y-axis tick labels
        y_label = ax.annotate(sel_col_2, xy=(0, 1), xytext=(-10, 10),
                            xycoords='axes fraction', textcoords='offset points',
                            ha='right', va='bottom', fontsize=12, rotation=0)

        # Calculate the space needed for the y-label
        renderer = fig.canvas.get_renderer()
        bbox = y_label.get_window_extent(renderer=renderer)
        text_width = bbox.width / fig.dpi  # Convert from pixels to inches

        # Ensure the left margin does not exceed a reasonable percentage of the figure width
        max_left_margin = 0.3 * fig.get_size_inches()[0]  # 30% of the figure width
        left_margin = min(text_width + 0.1, max_left_margin)  # Add some padding

        # Adjust the plot margins
        fig.subplots_adjust(left=left_margin / fig.get_size_inches()[0])

        st.pyplot(fig)
    
    elif (var_type_1 == 'Categorical' and var_type_2 == 'Numerical') or (var_type_1 == 'Numerical' and var_type_2 == 'Categorical'):
        # Create a grouped boxplot
        if var_type_1 == 'Categorical':
            cat_col, num_col = sel_col_1, sel_col_2
        else:
            cat_col, num_col = sel_col_2, sel_col_1

        
        fig, ax = plt.subplots()
        # df.boxplot(column=num_col, by=cat_col, ax=ax)
        # ax.set_xlabel(cat_col)
        # ax.set_ylabel(num_col)
        # st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.boxplot(x=cat_col, y=num_col, data=cleaned_df, color='lightgrey',
                    medianprops={'color': 'black', 'linewidth': 1}, # order=month_order,
                    flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 5,
                                'markeredgecolor': 'black'}, ax=ax)

        # ax.set_yticks([0, 25, 50, 75])
        # Set the labels
        ax.set_xlabel(cat_col)
        ax.set_ylabel('')
        # Add a custom y-label above the y-axis tick labels
        y_label = ax.annotate(num_col, xy=(0, 1), xytext=(-10, 10),
                            xycoords='axes fraction', textcoords='offset points',
                            ha='right', va='bottom', fontsize=12, rotation=0)

        # Calculate the space needed for the y-label
        renderer = fig.canvas.get_renderer()
        bbox = y_label.get_window_extent(renderer=renderer)
        text_width = bbox.width / fig.dpi  # Convert from pixels to inches

        # Ensure the left margin does not exceed a reasonable percentage of the figure width
        max_left_margin = 0.3 * fig.get_size_inches()[0]  # 30% of the figure width
        left_margin = min(text_width + 0.1, max_left_margin)  # Add some padding

        # Adjust the plot margins
        fig.subplots_adjust(left=left_margin / fig.get_size_inches()[0])

        ax.tick_params(axis='both', bottom=True, left=True, color='black',
                        width=1.0)

        sns.despine()

        st.pyplot(fig)

    else:
        st.write("I do not know what to do")