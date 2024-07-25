import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RANSACRegressor
from statsmodels.robust.scale import mad
from sklearn.linear_model import HuberRegressor
from statsmodels.api import RLM
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy.optimize import curve_fit
import os
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
from matplotlib.backend_tools import ToolBase, ToolToggleBase
import re
import math


def contains_word_from_list(string, word_list):
    for word in word_list:
        if word in string:
            return True
    return False


def valid_age_format(string):
    pattern = r'^\d+-\d+$'
    return bool(re.match(pattern, string))

def valid_quality_format(string):
    pattern = r'^0*(?:0\.\d+|1(?:\.0+)?)\s*-\s*0*(?:0\.\d+|1(?:\.0+)?)$'
    return bool(re.match(pattern, string))


def resize_barplot_ticks(self):
    #determine age range
    age_range = self.current_age_range
    age_limits = age_range.split('-')
    min_age = int(age_limits[0])
    max_age = int(age_limits[1])

    #we got a lot of values
    if(max_age - min_age > 60):
        #reduce fontsize
        self.ax.tick_params(axis='x', rotation=45, labelsize=3)
        #only show every second label
        for label in self.ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)


def search_optionsA(self, event, feature_list):
    # Get the current text in the Combobox
    current_text = self.filter_frame.featureA_combobox.get()

    if current_text == "" or current_text == "None":
        self.filter_frame.featureA_combobox['values'] = feature_list
    else:
        # Filter options based on the current text
        filtered_options = [option for option in feature_list if current_text.lower() in option.lower()]

        # Update the values of the Combobox with filtered options
        self.filter_frame.featureA_combobox['values'] = filtered_options


def search_optionsB(self, event, feature_list):
    # Get the current text in the Combobox
    current_text = self.filter_frame.featureB_combobox.get()

    if current_text == "" or current_text == "None":
        self.filter_frame.featureB_combobox['values'] = feature_list
    else:
        # Filter options based on the current text
        filtered_options = [option for option in feature_list if current_text.lower() in option.lower()]

        # Update the values of the Combobox with filtered options
        self.filter_frame.featureB_combobox['values'] = filtered_options





class FilterFrame(ttk.Frame):
    def __init__(self, parent, filter_command):
        super().__init__(parent, style='FilterFrame.TFrame')

        # Etiqueta para el rango de edad
        ttk.Label(self, text="Age Range:").grid(row=0, column=0, sticky="w")

        style = ttk.Style()
        style.configure('Custom.TFrame', background='light blue')

        frame = ttk.Frame(parent, style='Custom.TFrame')

        # Campo de entrada para el rango de edad
        self.age_entry = ttk.Entry(self)
        self.age_entry.grid(row=0, column=1, sticky="ew")
        self.age_entry.insert(0, "0-100")  # Valor por defecto

        
        

        # Etiqueta para el sexo
        ttk.Label(self, text="Sex Value:").grid(row=1, column=0, sticky="w")

        # Menú desplegable para la selección de sexo
        self.sex_combobox = ttk.Combobox(self, values=["M", "F", "Both", "M vs. F"])
        self.sex_combobox.grid(row=1, column=1, sticky="ew")
        self.sex_combobox.set("Both")  # Valor por defecto

        # Etiqueta para la columna que seleccionamos
        ttk.Label(self, text="Variable of Interest A: ").grid(row=2, column=0, sticky="w")
        ttk.Label(self, text="Variable of Interest B: ").grid(row=3, column=0, sticky="w")

        # Menú desplegable para la selección de la columna
        self.featureA_combobox = ttk.Combobox(self, values=["None"])
        self.featureA_combobox.grid(row=2, column=1, sticky="ew")
        self.featureA_combobox.set("None")  # Valor por defecto


        # Another Menu to choose variable of interests from (to compare with the first entry)
        self.featureB_combobox = ttk.Combobox(self, values=["None"])
        self.featureB_combobox.grid(row=3, column=1, sticky="ew")
        self.featureB_combobox.set("None")  # Valor por defecto

        # Botón para aplicar el filtro (Aquí debes vincular la lógica de filtrado)
        self.filter_button = ttk.Button(self, text="Apply Filter",
                                        command=lambda: filter_command(self.age_entry.get(), self.sex_combobox.get(),
                                                                       self.featureA_combobox.get(),
                                                                       self.featureB_combobox.get(),
                                                                       self.quality_entry.get()))
        self.filter_button.grid(row=4, column=0, columnspan=2, pady=5)

        # Configura el relleno de la fila y la columna
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

       
        # Agrega un espacio extra entre los Combobox de Feature of Interest y Quality
        ttk.Label(self, text="").grid(row=4, column=0, pady=(70, 10))

        # Etiqueta para el filtro de calidad
        ttk.Label(self, text="Quality:").grid(row=5, column=0, sticky="w")

        # Entrada para la selección de calidad
        self.quality_entry = ttk.Entry(self)
        self.quality_entry.insert(0, "0.1-1") 
        self.quality_entry.grid(row=5, column=1, sticky="ew", pady=(5, 0))
        

        

        

        

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.current_feature_of_interest = None
        self.feature_of_interest_data = None
        self.feature_of_interestB_data = None
        self.regression_type = None
        self.polynomial_features = None
        self.feature_of_interest = None
        self.feature_of_interestB = None
        self.deleted_df = pd.DataFrame()
        self.no_deleted_df = pd.DataFrame()
        self.title('Distribución de población por edad y sexo')
        self.features = None  # atributo para guardar los variables que nos interesan del CSV

        # En la clase App dentro del método __init__

        # Atributos para almacenar información de los filtros aplicados
        self.current_age_range = None
        self.current_sex_value = None
        self.current_feature_of_interest = None
        self.current_feature_of_interestB = None
        self.current_quality_value = None
        self.filtered_data = None  # Atributo para almacenar los datos filtrados

        self.configure(background='#e0e0e0')

        # Configura el peso de las columnas y filas para el redimensionamiento
        self.grid_columnconfigure(0, weight=0)  # Columna para los filtros
        self.grid_columnconfigure(1, weight=4)  # Columna para el área del gráfico y los botones
        self.grid_rowconfigure(0, weight=0)  # Fila para el gráfico
        self.grid_rowconfigure(1, weight=1)  # Fila para los botones

        self.dataframe = None

        # Sección de carga de CSV
        load_csv_button = ttk.Button(self, text="Load CSV", command=self.load_csv)
        load_csv_button.grid(row=0, column=0, columnspan=2, sticky='ew')  # Utiliza toda la anchura

        # Sección de filtro de datos
        self.filters_frame = ttk.LabelFrame(self, text="Data filter")
        self.filters_frame.grid(row=1, column=0, sticky='ns')

        self.filter_frame = FilterFrame(self.filters_frame, self.filter_data)
        self.filter_frame.pack(fill="x", expand=True)

        # Sección de gráfico
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().grid(row=1, column=1,
                                         sticky='nsew')  # Permite que el área del gráfico se expanda

        #Toolbar estandar de Pyplot, con algunas adaptaciones
        #self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        #self.toolbar = CustomNavigationToolbar(toolbar)
        self.toolbar.update()
        self.toolbar.grid(row = 2, column = 1)
        

        # Frame para los botones de gráfico y modelo
        self.buttons_frame = ttk.Frame(self)
        self.buttons_frame.grid(row=3, column=1, sticky='ew')  # Los botones quedan bajo el área del gráfico

        self.graph_type_combobox = ttk.Combobox(self.buttons_frame,
                                                values=["Histogram", "Bar Chart", "Line Chart", "Scatter"])
        self.graph_type_combobox.grid(row=0, column=0, padx=5, pady=5)
        self.graph_type_combobox.set("Histogram")

        self.regression_window_button = ttk.Button(self.buttons_frame, text="Fit Model", width=20,
                                                   command=self.create_regression_window)
        self.regression_window_button.grid(row=0, column=1, padx=5, pady=5)

        self.delete_outliers_button = ttk.Button(self.buttons_frame, text="Delete Outliers", width=20,
                                                 command=self.delete_outliers)
        self.delete_outliers_button.grid(row=0, column=2, padx=5, pady=5)

        self.export_data_button = ttk.Button(self.buttons_frame, text="Export Filtered Data", width=20,
                                             command=self.export_data)
        self.export_data_button.grid(row=1, column=1, padx=5, pady=5)

        self.export_outliers_button = ttk.Button(self.buttons_frame, text="Export Outliers", width=20,
                                                 command=self.export_outliers)
        self.export_outliers_button.grid(row=1, column=2, padx=5, pady=5)

        self.export_graph_button = ttk.Button(self.buttons_frame, text="Export Graph", width=20,
                                              command=self.export_graph)
        self.export_graph_button.grid(row=1, column=3, padx=5, pady=5)

        self.delete_outliers_button.config(state=tk.DISABLED)
        self.export_outliers_button.config(state=tk.DISABLED)
        self.export_graph_button.config(state=tk.DISABLED)
        self.regression_window_button.config(state=tk.DISABLED)

    def export_data(self):
        if not self.no_deleted_df.empty:
            df_orig = self.dataframe
            df2 = self.no_deleted_df
            rows_df_orig = df_orig[
                (df_orig[self.feature_of_interest].isin(df2['FoI'])) & (df_orig['Age'].isin(df2['Age']))]
        else:
            rows_df_orig = self.filtered_data
        # Diálogo para que el usuario pueda escoger en que formato guardar el archivo
        # Obtener la ruta de archivo para guardar los datos
        file_path = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[("Text files", '*.txt'), ("All Files", "*.*")],
            title="Save column as TXT...")

        # Verificar si se seleccionó un archivo
        if file_path:
            # Extraer la columna deseada
            column_name = 'Patient ID'
            column_to_save = rows_df_orig[column_name]

            # Guardar la columna en el archivo seleccionado
            column_to_save.to_csv(file_path, index=False, sep=' ', header=False)
            print(f"Datos filtrados guardados en {file_path}")
        else:
            print("Operación cancelada.")

    def create_regression_window(self):
        # Crear una nueva ventana
        regression_window = tk.Toplevel(self)
        regression_window.title("Select Regression Function")

        # Crear etiqueta y menú desplegable para seleccionar el tipo de regresión
        ttk.Label(regression_window, text="Select Regression Function:").pack()
        regression_combobox = ttk.Combobox(regression_window,
                                           values=["Exponential Regression", "Log Regression",
                                                   "Polynomial Regression Degree 2", "Polynomial Regression Degree 3",
                                                   "MC Regression", "Regression by Quartiles", "Huber Regression"])
        regression_combobox.pack()

        # Función para manejar la selección del usuario
        def apply_regression():
            selected_regression = regression_combobox.get()
            if selected_regression == "Exponential Regression":
                # Lógica para aplicar regresión lineal
                self.regression_type = "Exponential Regression"
            elif selected_regression == "Polynomial Regression Degree 3":
                # Lógica para aplicar regresión polinómica
                self.regression_type = "Polynomial Regression Degree 3"
            elif selected_regression == "Polynomial Regression Degree 2":
                # Lógica para aplicar regresión polinómica
                self.regression_type = "Polynomial Regression Degree 2"
            # Cierra la ventana después de seleccionar
            elif selected_regression == "Log Regression":
                self.regression_type = "Log Regression"
            elif selected_regression == "MC Regression":
                self.regression_type = "MC Regression"
            elif selected_regression == "Regression by Quartiles":
                self.regression_type = "Regression by Quartiles"
            elif selected_regression == "Huber Regression":
                self.regression_type = "Huber Regression"
            self.fit_model()
            regression_window.destroy()

        # Botón para aplicar la selección
        ttk.Button(regression_window, text="Apply", command=apply_regression).pack()

    def export_graph(self):
        if self.figure:
            # Diálogo para que el usuario pueda escoger en que formato guardar el gráfico
            file_path = filedialog.asksaveasfilename(
                defaultextension='.png',
                filetypes=[("PNG images", '*.png'), ("PDF files", '*.pdf'), ("SVG files", '*.svg'),
                           ("All Files", "*.*")],
                title="Save graph as...")
            if file_path:
                # Guarda la figura en el archivo seleccionado
                self.figure.savefig(file_path)
                print(f"Gráfico guardado en {file_path}")
            else:
                print("Operación cancelada.")
        else:
            print("No hay ningún gráfico para exportar.")


    def reg_MC(self, x, y):
        # Ajuste del modelo de regresión robusta por mínimos cuadrados
        self.model = RANSACRegressor()
        self.model.fit(x, y)

        # Predicciones del modelo
        y_fit = self.model.predict(x)

        # Calcular el error del modelo
        residuals = y - y_fit
        scale = 1.4826  # Factor de escala para convertir MAD a desviación estándar (para distribución normal)
        mad_residuals = mad(residuals)
        std_error = scale * mad_residuals

        # Límites del intervalo de confianza (por ejemplo, +/- 1.5 MAD)
        scale_factor = 1.5
        self.lower_limit = y_fit - scale_factor * std_error
        self.upper_limit = y_fit + scale_factor * std_error

        return x, y_fit

    def reg_quartiles(self, x, y):
        # Ajuste del modelo de regresión robusta por mínimos cuadrados
        self.model = RANSACRegressor()
        self.model.fit(x, y)

        # Predicciones del modelo
        y_fit = self.model.predict(x)

        # Calcular los residuos
        residuals = y - y_fit

        # Calcular los residuos ponderados (residuos absolutos)
        weighted_residuals = np.abs(residuals)

        # Utilizar más datos para calcular los cuartiles
        weighted_residuals_all = np.concatenate((weighted_residuals, weighted_residuals))

        # Calcular los cuantiles modificados de los residuos ponderados
        q1 = np.percentile(weighted_residuals_all, 25)
        q3 = np.percentile(weighted_residuals_all, 75)

        # Calcular los límites del intervalo de confianza basados en los cuartiles modificados
        iqr = q3 - q1
        scale_factor = 1.5
        self.lower_limit = y_fit - scale_factor * iqr
        self.upper_limit = y_fit + scale_factor * iqr

        # Regresar los resultados
        return x, y_fit

    def reg_huber(self, x, y):
        model = HuberRegressor()
        model.fit(x, y)

        # Predicciones del modelo
        y_fit = model.predict(x)

        # Si y es un DataFrame o Serie de Pandas, conviértelo a un array NumPy
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()

        # Calcular los residuos de Huber
        residuals = y - y_fit

        # Calcular la desviación estándar de los residuos
        std_error = np.std(residuals)

        # Límites del intervalo de confianza (por ejemplo, +/- 1.5 desviaciones estándar)
        scale_factor = 1.5
        self.lower_limit = y_fit - scale_factor * std_error
        self.upper_limit = y_fit + scale_factor * std_error

        return x, y_fit,

    def reg_polynomial(self, x, y, d):
        # Transformación de características polinómicas
        polynomial_features = PolynomialFeatures(degree=d)
        self.X_poly = polynomial_features.fit_transform(x)

        # Ajuste del modelo de regresión lineal
        self.model = LinearRegression()
        self.model.fit(self.X_poly, y)

        # Predicciones del modelo
        X_fit = np.linspace(min(x), max(x), 100).reshape(-1, 1)
        X_fit_poly = polynomial_features.transform(X_fit)
        y_fit = self.model.predict(X_fit_poly)

        # Calcular el error del modelo
        y_pred = self.model.predict(self.X_poly)
        mse = mean_squared_error(y, y_pred)
        std_error = np.sqrt(mse)  # Desviación estándar del error residual

        # Límites del intervalo de confianza (por ejemplo, +/- 1 desviación estándar)
        # Multiplicar la desviación estándar por un factor de escala
        scale_factor = d
        self.lower_limit = y_fit - scale_factor * std_error
        self.upper_limit = y_fit + scale_factor * std_error

        print(self.lower_limit)

        return X_fit, y_fit

    def reg_exponential(self, x, y):
        # Definir la función exponencial
        def exponential_func(x, a, b, c):
            return a * np.exp(b * x) + c

            # Inicializar los parámetros de la función exponencial
            p0 = (1.0, 1.0, 1.0)  # Valores iniciales de los parámetros (a, b, c)

            # Ajuste de la curva exponencial a los datos
            try:
                popt, pcov = curve_fit(exponential_func, x, y, p0=p0)
            except RuntimeError:
                # Si no se puede ajustar la curva exponencial, devolver None para indicar un fallo
                print("No se puede usar la funcion exponencial")
                return None, None

            # Predicciones del modelo
            y_pred = exponential_func(x, *popt)

            # Calcular el error del modelo
            mse = mean_squared_error(y, y_pred)
            std_error = np.sqrt(mse)  # Desviación estándar del error residual

            # Límites del intervalo de confianza (por ejemplo, +/- 1 desviación estándar)
            # Multiplicar la desviación estándar por un factor de escala
            scale_factor = 2
            self.lower_limit = exponential_func(x, *popt) - scale_factor * std_error
            self.upper_limit = exponential_func(x, *popt) + scale_factor * std_error

            return x, y_pred

    def reg_log(self, x, y):
        X_log = np.log(x)
        self.model = LinearRegression()
        self.model.fit(X_log.reshape(-1, 1), y)

        # Predicciones del modelo
        X_fit = np.linspace(min(x), max(x), 100).reshape(-1, 1)
        X_fit_log = np.log(X_fit)
        y_fit = self.model.predict(X_fit_log)

        # Calcular el error del modelo
        y_pred = self.model.predict(X_log.reshape(-1, 1))
        mse = mean_squared_error(y, y_pred)
        std_error = np.sqrt(mse)  # Desviación estándar del error residual

        # Límites del intervalo de confianza (por ejemplo, +/- 1 desviación estándar)
        # Multiplicar la desviación estándar por un factor de escala
        scale_factor = 2
        self.lower_limit = y_fit - scale_factor * std_error
        self.upper_limit = y_fit + scale_factor * std_error

        return X_fit, y_fit

    def fit_model(self):
        self.figure.clear()
        x = self.x
        y = self.feature_of_interest_data

        data = {'Age': self.x['Age'].values.flatten(), 'FoI': self.feature_of_interest_data.values.flatten()}
        df_data = pd.DataFrame(data)

        # Imputar los valores faltantes en X
        imputer = SimpleImputer(strategy='mean')
        x = imputer.fit_transform(x)

        label = None
        X_fit, y_fit = None, None
        if self.regression_type == "MC Regression":
            X_fit, y_fit = self.reg_MC(x, y)
            label = 'MCR'
        if self.regression_type == "Polynomial Regression Degree 2":
            X_fit, y_fit = self.reg_polynomial(x, y, 2)
            label = 'Polynomial Regression (degree {})'.format(2)
        if self.regression_type == "Polynomial Regression Degree 3":
            X_fit, y_fit = self.reg_polynomial(x, y, 3)
            label = 'Polynomial Regression (degree {})'.format(3)
        if self.regression_type == "Exponential Regression":
            self.reg_exponential(x, y)
            raise Exception("no se puede hacer")
            label = 'Exponential Regression'
        if self.regression_type == "Log Regression":
            X_fit, y_fit = self.reg_log(x, y)
            label = 'Log Regression'
        if self.regression_type == "Regression by Quartiles":
            X_fit, y_fit = self.reg_quartiles(x, y)
            label = 'Regression by Quartiles'
        if self.regression_type == "Huber Regression":
            X_fit, y_fit = self.reg_huber(x, y)
            label = 'Huber Regression'

        # Gráfico del modelo ajustado y líneas de error
        plt.scatter(x, y, color='blue', label='Data', s=2)
        plt.plot(X_fit, y_fit, color='red', label=label)

        X_fit = X_fit.flatten()
        lower_limit = self.lower_limit.flatten()
        upper_limit = self.upper_limit.flatten()

        plt.fill_between(X_fit, lower_limit, upper_limit, color='green', alpha=0.3, label='Acceptable Range')
        plt.xlabel('Age')
        plt.ylabel(self.feature_of_interest)
        plt.legend()

        self.figure.canvas.draw()
        self.delete_outliers_button.config(state=tk.NORMAL)
        # Crear un DataFrame con la edad y los límites de confianza
        limits_data = {'Age': X_fit.flatten(), 'Lower_Limit': lower_limit, 'Upper_Limit': upper_limit}
        df_limits = pd.DataFrame(limits_data)

        # Fusionar los DataFrames por la edad
        merged_df = pd.merge(df_data, df_limits, on='Age', how='inner')

        # Filtrar los datos dentro del rango de confianza
        self.no_deleted_df = merged_df[(merged_df['FoI'] >= merged_df['Lower_Limit']) &
                                       (merged_df['FoI'] <= merged_df['Upper_Limit'])]
        if self.deleted_df.empty:
            self.deleted_df = merged_df[(merged_df['FoI'] >= merged_df['Lower_Limit']) &
                                        (merged_df['FoI'] <= merged_df['Upper_Limit'])]
        else:
            deleted_df2 = merged_df[(merged_df['FoI'] >= merged_df['Lower_Limit']) &
                                    (merged_df['FoI'] <= merged_df['Upper_Limit'])]
            self.deleted_df = pd.concat([self.deleted_df, deleted_df2], ignore_index=True)

    def delete_outliers(self):

        # Obtener las coordenadas X e Y de los puntos dentro del rango de confianza
        filtered_x = self.no_deleted_df['Age']
        filtered_y = self.no_deleted_df['FoI']

        self.figure.clear()
        # Actualizar el gráfico con los puntos dentro del rango de confianza
        plt.scatter(filtered_x, filtered_y, color='blue', label='Data (Within Confidence Interval)', s=2)
        plt.xlabel('Age')
        plt.ylabel(self.feature_of_interest)
        plt.legend()

        # Mostrar el gráfico actualizado
        self.figure.canvas.draw()

        self.feature_of_interest_data = (filtered_y.to_frame())
        self.x = filtered_x.to_frame()
        self.export_outliers_button.config(state=tk.NORMAL)

    def export_outliers(self):

        if not self.deleted_df.empty:
            df_orig = self.dataframe
            df2 = self.deleted_df
            rows_df_orig = df_orig[
                (df_orig[self.feature_of_interest].isin(df2['FoI'])) & (df_orig['Age'].isin(df2['Age']))]

            # Diálogo para que el usuario pueda escoger en que formato guardar el archivo
            # Obtener la ruta de archivo para guardar los datos
            file_path = filedialog.asksaveasfilename(
                defaultextension='.txt',
                filetypes=[("Text files", '*.txt'), ("All Files", "*.*")],
                title="Save column as TXT...")

            # Verificar si se seleccionó un archivo
            if file_path:
                # Extraer la columna deseada
                column_name = 'Patient ID'
                column_to_save = rows_df_orig[column_name]

                # Guardar la columna en el archivo seleccionado
                column_to_save.to_csv(file_path, index=False, sep=' ', header=False)
                print(f"Datos anomalos guardados en {file_path}")
            else:
                print("Operación cancelada.")


    
    

    def load_csv(self):
        



        #FILEPATH FOR TESTING
        cwd = os.getcwd()
        file_path = cwd + "/volbrain_data.parquet"  

        #if the file is at a specific place the code below can be commented out and 
        #the filepath can be specified here
        #filepath = "/PATH/TO/PARQUET"


        #CODE FOR RELEASE
        """
        file_path = filedialog.askopenfilename(filetypes=[("Parquet files", "*.parquet")])
        if file_path:
            absolute_path = os.path.abspath(file_path)
        else:
            messagebox.showerror("Error","No se ha seleccionado un fichero")
            return 
        
        file_path.encode('unicode_escape')  
        """

        try:
            self.dataframe = pd.read_parquet(file_path, engine='fastparquet')
            

            # crear menu desplegable con todos los valores que aparecen en las columnas del fichero (con filtros de palabras clave)
            features = ["None"]
            for column_name, column_entries in self.dataframe.items():

                # este filtro se puede adaptar
                if contains_word_from_list(column_name.lower(),
                                           ['asymmetry', "cm3", "thickness", "total", "count", "lesion"]):
                    features.append(column_name)

            # actualizar el boton deplegable
            self.filter_frame.featureA_combobox['values'] = features
            self.filter_frame.featureB_combobox['values'] = features

            # guardar los features
            self.features = features

            # hace que el menu desplegable se puede buscar
            self.filter_frame.featureA_combobox.bind('<KeyRelease>',
                                                   lambda event: search_optionsA(self, event, self.features))
                                                
            self.filter_frame.featureB_combobox.bind('<KeyRelease>',
                                                   lambda event: search_optionsB(self, event, self.features))                            
                                                 

        except Exception as e:
            messagebox.showerror("Error",f"Error loading file: {e}")


    

    # Método para actualizar el gráfico con los datos
    def update_plot(self, data, graph_type="Histogram"):
        feature_of_interest = self.current_feature_of_interest
        feature_of_interestB = self.current_feature_of_interestB

        # si no hay un dataframe cargado se sale de la funcion
        if not hasattr(self, 'dataframe'):
            messagebox.showerror("Error", "Carga un fichero CSV primero")
            return

        self.ax.clear()

        # Mejorar la visibilidad
        tick_label_fontsize = 5
        tick_label_rotation = 45



        #set the labels here, so that that they can be adapted for specfici cases in if statement below, same goes for tick_params
        self.ax.set_xlabel('Edad' if 'Age' in data else 'Age', fontsize=7, weight='bold')
        self.ax.set_ylabel('Cantidad' if feature_of_interest is None else feature_of_interest, fontsize=7,
                           weight='bold')

        # Aplicar configuraciones de estilo para los ticks de los ejes
        self.ax.tick_params(axis='x', labelsize=tick_label_fontsize, labelrotation=tick_label_rotation,
                            )
        self.ax.tick_params(axis='y', labelsize=tick_label_fontsize)
       

        #We are plotting values just by Males, Females or the aggregation of both, NOT comparing data
        if(self.current_sex_value != 'M vs. F'):

           
            if (feature_of_interest is None) and (feature_of_interestB is not None):
                messagebox.showerror("Error", "If you just want to plot one feature, select it in the box for feature A")
                return

            
            #we are not analizing a specific feature, we are just looking at the number of entries for each age
            elif (feature_of_interest is None) and (feature_of_interestB is None):
                data.dropna(subset=['Age'], inplace=True)

                #Perform Age Counts to get the number of Men and Women in each Age Category
                age_counts = data['Age'].value_counts().sort_index()

                if graph_type == "Histogram":
                    self.ax.hist(data['Age'], bins=range(int(data['Age'].min()), int(data['Age'].max()) + 1, 1))

                
                elif graph_type == "Bar Chart":
                    #self.ax.bar(age_counts.index, age_counts.values)
                    counts_df = pd.DataFrame(list(age_counts.items()), columns=['Age', 'Count'])

            

                    #Plot the data of the merged df in barchart and resize if needed
                    sns.barplot(data=counts_df, x='Age', y='Count')
                    resize_barplot_ticks(self)

                elif graph_type == "Line Chart":
                    self.ax.plot(age_counts.index, age_counts.values)
                elif graph_type == "Scatter":
                    counts_df = pd.DataFrame(list(age_counts.items()), columns=['Age', 'Count'])
                    sns.scatterplot(data=counts_df, x='Age', y='Count', s=10, ax=self.ax)


            #plot for !one! specific feature
            elif (feature_of_interest is not None) and (feature_of_interestB is None):
                data.dropna(subset=[feature_of_interest], inplace=True)

                if graph_type == "Histogram":
                    self.ax.hist(data[feature_of_interest], bins=20)
                    self.ax.set_ylabel('Cantidad')
                    self.ax.set_xlabel(feature_of_interest)
                elif graph_type == "Bar Chart":
                    sns.barplot(data=data, x='Age', y=feature_of_interest, ax=self.ax)
                    
                    #handle the sizing of the tick lables if the age range is big 
                    resize_barplot_ticks(self)
                elif graph_type == "Line Chart":
                    mean_features = data.groupby(['Age'])[feature_of_interest].mean().reset_index()
                    self.ax.plot(mean_features['Age'], mean_features[feature_of_interest])
                elif graph_type == "Scatter":
                    sns.scatterplot(data=data, x='Age', y=feature_of_interest, s=10, ax=self.ax)
                    self.regression_window_button.config(state=tk.NORMAL)
            
            
            #plot two features one vs the other 
            else:
                #Drop NaNs in both columns
                data.dropna(subset=[feature_of_interest], inplace=True)
                data.dropna(subset=[feature_of_interestB], inplace=True)



                if graph_type == "Histogram":
                    self.ax.hist([data[feature_of_interest],data[feature_of_interestB]], bins=20, label=[feature_of_interest,feature_of_interestB])
                    self.ax.set_ylabel('Cantidad')
                    self.ax.set_xlabel(f'{feature_of_interest} and {feature_of_interestB}')



                elif graph_type == "Bar Chart":
                    plot_df = pd.melt(data, id_vars=['Age'], value_vars=[feature_of_interest, feature_of_interestB], 
                    var_name='FeatureType',value_name='Features')

                    
                    sns.barplot(data=plot_df, x='Age', y='Features', hue='FeatureType',ax=self.ax)
                    self.ax.set_xlabel(f'{feature_of_interest} and {feature_of_interestB}')
        
                    #handle the sizing of the tick lables if the age range is big 
                    resize_barplot_ticks(self)
                elif graph_type == "Line Chart":
                    featureA_means = data.groupby(['Age'])[feature_of_interest].mean().reset_index()
                    featureB_means = data.groupby(['Age'])[feature_of_interestB].mean().reset_index()

            
                    self.ax.plot(featureA_means['Age'],featureA_means[feature_of_interest], label=feature_of_interest)
                    self.ax.plot(featureB_means['Age'],featureB_means[feature_of_interestB], label=feature_of_interestB)
                
                elif graph_type == "Scatter":
                    plot_df = pd.melt(data, id_vars=['Age'], value_vars=[feature_of_interest, feature_of_interestB], 
                    var_name='FeatureType',value_name='Features')

                    
                    sns.scatterplot(data=plot_df, x='Age', y='Features', s=10,hue='FeatureType', ax=self.ax)
                    

                self.ax.legend(loc='upper right', prop={'size': 6})


        elif(feature_of_interestB is not None):
           messagebox.showerror("Error", "For Feature B \"None\" has to be selected, if you want to compare Men vs Women, you can also select \"None\" for both features to study the number of samples for each Age group") 
           return
        



        #we are comparing female vs male entries
        else:
              
            # if we just want to now the count of men vs women
            if feature_of_interest is None:
                data.dropna(subset=['Age'], inplace=True)

                #create separate DFs for men and women
                men_df = data[data['Sex']== 'Male']
                women_df = data[data['Sex']=='Female']  


                #Perform Age Counts to get the number of Men and Women in each Age Category, 
                # create new df to store that information
                men_age_counts = men_df['Age'].value_counts().sort_index()
                women_age_counts = women_df['Age'].value_counts().sort_index()

                # Create DataFrame for male & female counts
                men_counts_df = pd.DataFrame({'Sex': ['Male'] * len(men_age_counts), 'Age': list(men_age_counts.index), 'Count': list(men_age_counts.values)})
                women_counts_df = pd.DataFrame({'Sex': ['Female'] * len(women_age_counts), 'Age': list(women_age_counts.index), 'Count': list(women_age_counts.values)})

                # Concatenate DataFrames
                merged_df = pd.concat([men_counts_df, women_counts_df], ignore_index=True)

                
                
                if graph_type == "Histogram":
                    self.ax.hist([men_df['Age'],women_df['Age']], bins=range(int(data['Age'].min()), int(data['Age'].max()) + 1, 1), label=['Male','Female'])
                    
                elif graph_type == "Bar Chart":
                    #Plot the data of the merged df in barchart and resize if needed
                    sns.barplot(data=merged_df, x='Age', y='Count', hue='Sex')
                    
                    resize_barplot_ticks(self)

                elif graph_type == "Line Chart":
                    self.ax.plot(men_counts_df['Age'], men_counts_df['Count'], label='Male')
                    self.ax.plot(women_counts_df['Age'], women_counts_df['Count'], label='Female')
                    
                elif graph_type == "Scatter":
                    sns.scatterplot(data=merged_df, x='Age', y='Count', hue='Sex', s=10, ax=self.ax)
                    

                self.ax.legend(loc='upper right', prop={'size': 6})


            #plot for a specific feature
            else:
                data.dropna(subset=[feature_of_interest], inplace=True)
                
                
                #create separate DFs for men and women
                men_df = data[data['Sex']== 'Male']
                women_df = data[data['Sex']=='Female'] 
                
                if graph_type == "Histogram":
                    self.ax.hist([men_df[feature_of_interest],women_df[feature_of_interest]], bins=20, label=['Male','Female'])
                    self.ax.set_ylabel('Cantidad')
                    self.ax.set_xlabel(feature_of_interest)
                    

                elif graph_type == "Bar Chart":
                    sns.barplot(data=data, x='Age', y=feature_of_interest, hue='Sex',ax=self.ax)
                    
        
                    #handle the sizing of the tick lables if the age range is big 
                    resize_barplot_ticks(self)

                elif graph_type == "Line Chart":
                    men_mean_features = men_df.groupby(['Age'])[feature_of_interest].mean().reset_index()
                    women_mean_features = women_df.groupby(['Age'])[feature_of_interest].mean().reset_index()


                    self.ax.plot(men_mean_features['Age'], men_mean_features[feature_of_interest], label='Male')
                    self.ax.plot(women_mean_features['Age'], women_mean_features[feature_of_interest], label='Female')
                    

                elif graph_type == "Scatter":
                    sns.scatterplot(data=data, x='Age', y=feature_of_interest, s=10,hue='Sex', ax=self.ax)

                
                
                
                self.ax.legend(loc='upper right', prop={'size': 6})        


        



        
        
        self.figure.tight_layout()
        self.figure.canvas.draw()

    def filter_data(self, age_range, sex_value, feature_of_interest, feature_of_interestB, quality_range):
        #check if the features that we typed in actually exist
        if(feature_of_interest not in self.features):
            messagebox.showerror("Error", "This feature does not exist, you can only choose between the valid features for Feature A")
            return

        if(feature_of_interestB not in self.features):
            messagebox.showerror("Error", "This feature does not exist, you can only choose between the valid features for Feature B")
            return

       

        # If feature of interest is the string none, convert it to keyword None
        if (feature_of_interest == "None"):
            feature_of_interest = None
        # else convert the number to numeric, also check if the column has any entries at all
        else:
            self.dataframe[feature_of_interest] = pd.to_numeric(self.dataframe[feature_of_interest], errors='coerce')

            #check if there are any numbers in the column
            presence_of_num_series = self.dataframe[feature_of_interest].apply(lambda x: isinstance(x,float) and not math.isnan(x))
            
            
            

            #inverse the previous series, to use pandas all() function and determine if the column contains ANY number
            no_num_in_column = False
            
            
            if((~presence_of_num_series).all()):
                no_num_in_column = True
            
            if(no_num_in_column):
                messagebox.showerror("Error", "The selected feature column of Feature A has no entries")
                return



        #Repeat the same process for the feature of interest B
        # If feature of interest is the string none, convert it to keyword None
        if (feature_of_interestB == "None"):
            feature_of_interestB = None
        # else convert the number to numeric, also check if the column has any entries at all
        else:
            self.dataframe[feature_of_interestB] = pd.to_numeric(self.dataframe[feature_of_interestB], errors='coerce')

            #check if there are any numbers in the column
            presence_of_num_series = self.dataframe[feature_of_interestB].apply(lambda x: isinstance(x,float) and not math.isnan(x))
            
        
            #inverse the previous series, to use pandas all() function and determine if the column contains ANY number
            no_num_in_column = False
            
            
            if((~presence_of_num_series).all()):
                no_num_in_column = True
            
            if(no_num_in_column):
                messagebox.showerror("Error", "The selected feature column of Feature B has no entries")
                return


        # Convertir Age y QC a numéricos con errores coercitivos
        self.dataframe['Age'] = pd.to_numeric(self.dataframe['Age'], errors='coerce')

        if 'QC' in self.dataframe.columns:
            self.dataframe['QC'] = pd.to_numeric(self.dataframe['QC'], errors='coerce')


       



        # Iniciar el DataFrame filtrado
        filtered_df = self.dataframe

      
        ###### Filtrar por calidad
        if not valid_quality_format(quality_range):
            messagebox.showerror("Error", "Quality tiene que ser en el formato float-float y los valores tienen que ser en el rango entre 0 y 1 , p.ej. 0.2-0.7.")
            return

        quality_limits = quality_range.split('-')
        min_quality = float(quality_limits[0])
        max_quality = float(quality_limits[1])

        if min_quality > 1 or max_quality > 1:
            messagebox.showerror("Error", "Los valores de quality tienen que ser menor que 1")
            return 

        if min_quality >= max_quality:
            messagebox.showerror("Error", "La calidad mínima debería ser menor que la calidad máxima.")
            return
    
        #filter according to quality range
        filtered_df = filtered_df[(filtered_df['QC'] >= min_quality) & (filtered_df['QC'] <= max_quality)]

        
        ###### Aplicar filtro de edad
        #check that age range is in valid format, if not exit the function
        if not valid_age_format(age_range):
            messagebox.showerror("Error", "Edad tiene que ser en el formato numero-numero, p.ej. 0-80")
            return
            
        age_limits = age_range.split('-')
        min_age = int(age_limits[0])
        max_age = int(age_limits[1])

        #check that the boundaries are set reasonably
        if min_age >= max_age:
            messagebox.showerror("Error", "La edad mínima debería ser menor que la edad máxima")
            return

        filtered_df = filtered_df[(filtered_df['Age'] >= min_age) & (filtered_df['Age'] <= max_age)]


    
        ###### Aplicar filtro de sexo
        if sex_value != 'Both' and sex_value != 'M vs. F':
            sex_value = 'Male' if sex_value == 'M' else 'Female'
            filtered_df = filtered_df[filtered_df['Sex'] == sex_value]

        # Guardar los filtros actuales en los atributos de la clase
        self.current_age_range = age_range
        self.current_sex_value = sex_value
        self.current_feature_of_interest = feature_of_interest
        self.current_feature_of_interestB = feature_of_interestB
        self.current_quality_range = quality_range

        # Actualizar el gráfico y guardar los datos filtrados
        self.update_plot(filtered_df, self.graph_type_combobox.get())
        self.filtered_data = filtered_df
        if feature_of_interest != None:
            self.feature_of_interest_data = filtered_df[feature_of_interest].to_frame()

        if feature_of_interestB != None:
            self.feature_of_interestB_data = filtered_df[feature_of_interestB].to_frame()

        
        self.x = filtered_df['Age'].to_frame()

        self.feature_of_interest = feature_of_interest
        self.feature_of_interestB = feature_of_interestB

        
        

        self.filtered_data = filtered_df

        
# Ejecutar la aplicación
if __name__ == '__main__':
    app = App()
    app.load_csv()
    app.mainloop()
