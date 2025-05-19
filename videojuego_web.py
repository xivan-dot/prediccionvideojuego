# Importar librerías
import streamlit as st
import pickle
import pandas as pd
import sklearn

# Configuración de la página (debe ser la primera instrucción)
############################################################################################################################
# Título principal centrado
st.set_page_config(page_title="Modelo para la predicción de compra de videojuengos en tienda", layout="centered")

st.title("Predicción para la compra de videojuegos en la tienda")
# Cambiar la fuente de toda la aplicación
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Impact', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, button, input, select, textarea {
        font-family: 'Impact', sans-serif !important;
    }

    .stButton>button, .stSelectbox, .stSlider {
        font-family: 'Impact', sans-serif !important;
    }

    table, th, td {
        font-family: 'Impact', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)


# montar imagen

st.image("tienda.jpg")

#Cargamos el modelo
import pickle
filename = 'modelo-reg-tree-knn-nn1.pkl'
#filename = 'modelo-reg-tree.pkl'
model_Tree,model_Knn, model_NN,variables, min_max_scaler = pickle.load(open(filename, 'rb')) #DT-Knn

#Creamos el sidebar
st.sidebar.title("Parametros del comprador")

def main():

    #Creamos las entradas del modelo

    def user_input_feature():
        edad = st.sidebar.number_input("Edad", 14, 52)

        option_juego = ['F1', 'Dead space', 'Sim City', 'Battlefield', 'Fifa', 'Mass Effect', 'KOA: Reckoning']
        videojuego = st.sidebar.selectbox('Seleccione el tipo de videojuego que quiere comprar', option_juego)

        option_plataforma = ['Play Station', 'PC', 'Xbox', 'Otros', 'mobile']
        plataforma = st.sidebar.selectbox('Plataforma usada por el videojuego', option_plataforma)

        option_sex = ['macho pecho pelu', 'Mujer']
        sexo = st.sidebar.selectbox('Sexo del comprador', option_sex)

        Consumidor_habitual = st.sidebar.checkbox('Consume videojuegos habitualmente', value=False)

        data = {
            "Edad": edad,
            "videojuego": videojuego,
            "Plataforma": plataforma,
            "Sexo": sexo,
            "Consumidor_habitual": Consumidor_habitual
        }

        feature = pd.DataFrame([data])  # Creamos el DataFrame correctamente
        return feature

    features = user_input_feature() #  permite ver en el front el sidebar
    

     # Preparar los datos
    data_preparada = features.copy()

    # Crear las variables dummies
    data_preparada = pd.get_dummies(data_preparada, columns=['videojuego', 'Plataforma'], drop_first = False)
    #data_preparada
    data_preparada = pd.get_dummies(data_preparada, columns=['Sexo', 'Consumidor_habitual'], drop_first = False)# se elimina una dummy porque solo tiene 2 categorias
              
    
    #Se adicionan las columnas faltantes

    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)# Si falta una variable la crea y llena con ceros
    st.subheader("Datos que ingresan modelo para realizar la prediccion")
    data_preparada
    
    # Realizar predicción con NN por ser el mejor modelo recordar que hay que aplicar minmaxSacaler
    data_preparada[['Edad']]= min_max_scaler.transform(data_preparada[['Edad']])
    #st.write("datos variables del modelo con minmaxSacaler") # Revisión de aplicación de la normaliación
    #data_preparada

    if st.button('Realizar Predicción'):
        y_fut = model_NN.predict(data_preparada)
        
        st.success(f'La predicción es: {y_fut[0]:.1f} dólares') #y_fut[0]: Extrae el primer (y único) valor de la lista. :.1f: Formatea el número con una sola cifra decimal. dólares: Añade la palabra después del número.

        

if __name__ == '__main__':
    main()
