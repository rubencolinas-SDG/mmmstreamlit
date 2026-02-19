import pandas as pd
import streamlit as st #type:ignore
from st_aggrid import AgGrid, GridOptionsBuilder # type: ignore
from streamlit_extras.stylable_container import stylable_container # type: ignore
from main.config import Colors
    
  
# Función para aplicar estilo a la tabla
def style_table() -> str:
    #Aplicamos estilos
    css = {
        #Estilos del Header
        ".ag-header-cell": {
            "background-color": Colors.CUPRA_COPPER,
            },
        ".ag-header-cell-text":{
            "font-size":"12px", 
            "font-weight": "bold",
        },
        ".ag-header-cell-label":{
            "color":"white",
            "justify-content":"center",
            "display":"flex",
            "width":"100%"
        },
        #Estilo del cuerpo
        ".ag-cell":{
            "text-align":"center",
            "font-size":"12px"
        }
        
    }

    return css

# Función que devuelve la estructura de la tabla
def make_structure_table() -> str:
    
   

    # Primera columna
    custom_defs = [
        { 
            "headerName": "Modelo", 
            "field": "Modelo", 
            "width": 250, 
            "checkboxSelection": True, 
            "headerCheckboxSelection": True, 
            "suppressMenu": True ,
            "cellStyle": {'textAlign': 'left', 'fontSize':'10px'}
        },
        { 
            "headerName": "Tipo Motor", 
            "field": "Motorización", 
            "width": 150, 
            "suppressMenu": True ,
            "cellStyle": {'textAlign': 'center', 'fontSize':'10px'}
        },
        { 
            "headerName": "Precio Vehiculo(€)", 
            "field": "Precio", 
            "width": 150, 
            "suppressMenu": True ,
            "cellStyle": {'textAlign': 'center', 'fontSize':'10px'}
        }
    ]

    

    return custom_defs

def show() -> None: 

    # Obtenemos los datos
    data = pd.DataFrame({
        "Modelo": ["CUPRA Formentor", "SEAT León", "CUPRA Born", "SEAT Ibiza", "CUPRA Ateca"],
        "Motorización": ["VZ5", "e-Hybrid", "Eléctrico 58kWh", "TSI 110 CV", "2.0 TSI"],
        "Precio": [65000, 28000, 39000, 19000, 45000]
    })

    # Generamos el GridOptionsBuilder
    gb = GridOptionsBuilder.from_dataframe(data)

    # Configuramos el Header
    gb.configure_grid_options(
        headerHeight=20, # Altura Header
        rowHeight=20 # Altura Fila
    )

    # Configuramos las columnas
    gb.configure_default_column(
        resizable=True,
        filterable=False,
        sortable=True
    )

    # Configuramos la seleccion
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)

    gridOptions = gb.build()

    # Obtenemos la estructura de la tabla
    structure = make_structure_table()
    gridOptions['columnDefs'] = structure

    AgGrid(    
        data, 
        gridOptions=gridOptions, 
        allow_unsafe_jscode=True, 
        theme="alpine", 
        height=180,
        custom_css=style_table(),
        fit_columns_on_grid_load=False
    )