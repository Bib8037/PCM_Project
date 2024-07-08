import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

def main():
    st.title("Regression Model Generator")

    # File upload
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Ensure 'Date' is in datetime format
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        # Filter the data where 'PCM Install' == 1
        st.write(df.columns)
        df_filtered = df[df['PCM Install'] == 1]

        # Column selection
        st.header("Select X and Y Columns")
        x_columns = st.multiselect("Select X columns", [col for col in df.columns if col not in ['Date', 'PCM Install']])
        y_column = st.selectbox("Select Y column", df.columns)

        # Polynomial feature option
        st.header("Polynomial Features")
        polynomial_degree = st.selectbox("Select polynomial degree", [1, 2, 3, 4], index=0)
        
        if len(x_columns) > 0 and y_column:
            # Polynomial features transformation
            if polynomial_degree > 1:
                poly = PolynomialFeatures(degree=polynomial_degree)
                X_poly = poly.fit_transform(df_filtered[x_columns])
                X_poly_full = poly.fit_transform(df[x_columns])
                feature_names = poly.get_feature_names_out(x_columns)
            else:
                X_poly = df_filtered[x_columns]
                X_poly_full = df[x_columns]
                feature_names = x_columns

            # Model generation
            st.header("Regression Model")
            model = LinearRegression()
            model.fit(X_poly, df_filtered[y_column])

            intercept = model.intercept_
            coefficients = model.coef_

            # Display the linear regression formula
            formula = f"{y_column} = {intercept:.2f}"
            for coef, col in zip(coefficients, feature_names):
                formula += f" + ({coef:.2f} * {col})"
            st.write("Linear Regression Formula:")
            st.latex(formula)

            st.write("Intercept:", intercept)
            st.write("Coefficients:", dict(zip(feature_names, coefficients)))

            # Model evaluation
            st.header("Model Evaluation")
            train_score = model.score(X_poly, df_filtered[y_column])
            st.write("Train R2 Score:", train_score)

            # Splitting the data into train and test sets
            st.header("Train-Test Split")
            train_size = st.slider("Train set size", min_value=0.1, max_value=0.9, value=0.7, step=0.1)
            train_df, test_df = train_test_split(df_filtered, train_size=train_size, random_state=42)

            if polynomial_degree > 1:
                X_poly_train = poly.transform(train_df[x_columns])
                X_poly_test = poly.transform(test_df[x_columns])
            else:
                X_poly_train = train_df[x_columns]
                X_poly_test = test_df[x_columns]

            # Model training on train set
            model.fit(X_poly_train, train_df[y_column])

            # Model evaluation on test set
            test_score = model.score(X_poly_test, test_df[y_column])
            st.write("Test R2 Score:", test_score)

            # Predictions on all data
            df['y_pred'] = model.predict(X_poly_full)

            # Mean Squared Error (MSE)
            mse = mean_squared_error(test_df[y_column], model.predict(X_poly_test))
            st.write("Mean Squared Error (MSE):", mse)

            # Mean Absolute Error (MAE)
            mae = mean_absolute_error(test_df[y_column], model.predict(X_poly_test))
            st.write("Mean Absolute Error (MAE):", mae)

            # Sort data by Date for plotting
            df_sorted = df.sort_values(by='Date')

            # Interactive line graph of actual y and predicted y
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_sorted['Date'], y=df_sorted[y_column], mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(x=df_sorted['Date'], y=df_sorted['y_pred'], mode='lines+markers', name='Predicted'))
            fig.update_layout(title='Actual vs Predicted Energy Consumption',
                              xaxis_title='Date',
                              yaxis_title=y_column)
            st.plotly_chart(fig)

            # Add bar chart for monthly actual vs predicted
            df_sorted['Month'] = df_sorted['Date'].dt.to_period('M').astype(str)
            monthly_summary = df_sorted.groupby('Month')[[y_column, 'y_pred']].sum().reset_index()
            bar_fig = px.bar(monthly_summary, x='Month', y=[y_column, 'y_pred'], barmode='group',
                             title='Monthly Actual vs Predicted Energy Consumption')
            st.plotly_chart(bar_fig)

            # Add table to summarize differential between actual and predicted
            monthly_summary['Energy Saving'] = monthly_summary[y_column] - monthly_summary['y_pred']
            st.write("Monthly Summary of Actual vs Predicted Energy Consumption")
            st.dataframe(monthly_summary)

if __name__ == "__main__":
    main()
