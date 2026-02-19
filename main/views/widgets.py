import streamlit as st
from main.components.label import Label
from main.config import Colors
from millify import millify
import streamlit_antd_components as sac
from streamlit_extras.stylable_container import stylable_container
import main.components.table_example as table_example

def show_view() -> None:
    
    # Componente para escribir texto personalizado
    Label(
        id="title_widgets",
        text="Componente Para Escribir Texto",
        font_size=16,
        text_color=Colors.CUPRA_COPPER,
        font_weight="bold"
    ).render()

    # Componente para dar formato a los numeros
    st.markdown("")
    numero = 12345678
    numero_formateado = millify(numero)
    numero_decimales = millify(numero, precision=2)
    st.text(f"Número sin formato: {numero}")
    st.text(f"Número con formato: {numero_formateado}")
    st.text(f"Número con formato y decimales: {numero_decimales}")

    # Botones personalizado
    with stylable_container(
    key="boton_personalizado",
    css_styles=f"""
        button {{
            background-color: {Colors.CUPRA_COPPER};
            color: {Colors.WHITE};
            border-radius: 20px;
        }}
        button:hover {{
            background-color: {Colors.DESSERT_SAND} !important;
            color: {Colors.PITCH_BLACK} !important;
            border-color: {Colors.PITCH_BLACK} !important;
        }}
        """
    ):
        if st.button("Botón Personalizado"):
            st.write("¡Click recibido!")


    # Tabla
    st.markdown("")
    table_example.show()