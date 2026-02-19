import streamlit as st
import base64

# Clase para generar imagenes personalizadas en Streamlit.

class Image:
   
   # Constructor de la clase Image
    def __init__(self, id, url, local, width,border_radius=0):
        self.id = id
        self.url = url
        self.local = local
        self.width = width
        self.border_radius = border_radius


        
    # Método para renderizar la etiqueta con los estilos definidos
    def render(self):

        if self.local:
            # Si es una imagen local, la cargamos directamente
            # Convertir imagen a base64
            with open(self.url, "rb") as img_file:
                img_bytes = img_file.read()
                img_base64 = base64.b64encode(img_bytes).decode()


            # Generamos la etiqueta HTML para la imagen
            st.markdown(f"""
                    <style>
                    .custom-img-{self.id} {{
                        width: {self.width}px;
                        border-radius: {self.border_radius}px;
                    }}

                    .no-margin {{
                        margin: 0 !important;
                        padding: 0 !important;
                    }}
                    </style>

                    <div class=".no-margin">
                        <img class="custom-img-{self.id}" src="data:image/jpeg;base64,{img_base64}">
                    </div>

                """, unsafe_allow_html=True
            )
        else:
            # Mostrar imagen desde URL directamente
            st.markdown(f"""
                    <style>
                    .custom-img-{self.id} {{
                        width: {self.width}px;
                        border-radius: {self.border_radius}px;
                    }}

                    .no-margin {{
                        margin: 0 !important;
                        padding: 0 !important;
                    }}
                    </style>

                    <div class=".no-margin">
                        <img class="custom-img-{self.id}" src="{imagen}">
                    </div>          
                """, unsafe_allow_html=True
            )