import streamlit as st  # type: ignore
from main.config import Colors

# Clase para generar etiquetas personalizadas en Streamlit.
# Permite crear etiquetas con texto y estilos personalizados.


class Label:

    # Constructor de la clase Label
    def __init__(
        self,
        id,
        text,
        font_size,
        text_color=Colors.PITCH_BLACK,
        font_weight="normal",
        margin_left=0,
        margin_right=0,
        text_align="center",
        margin_top=0,
        margin_bottom=0,
        padding_top=0,
        padding_bottom=0,
        tag="label",
    ):
        self.id = id
        self.text = text
        self.font_size = font_size
        self.text_color = text_color
        self.font_weight = font_weight
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.text_align = text_align
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.padding_top = padding_top
        self.padding_bottom = padding_bottom
        self.tag = tag

    # Método para renderizar la etiqueta con los estilos definidos
    def render(self):
        st.markdown(
            f"""
            <style>
            .custom-label-{self.id} {{
                font-size: {self.font_size}px;
                color: {self.text_color};
                font-weight: {self.font_weight};
                text-align: {self.text_align};;
                margin-left: {self.margin_left}px;
                margin-right: {self.margin_right}px;
                margin-top: {self.margin_top};
                margin-bottom: {self.margin_bottom};
                padding-top: {self.padding_top};
                padding-bottom: {self.padding_bottom};
            }}

            
            </style>
            <div class="no-margin">
                <{self.tag} class="custom-label-{self.id}">{self.text}</{self.tag}>
            </div>
        """,
            unsafe_allow_html=True,
        )
