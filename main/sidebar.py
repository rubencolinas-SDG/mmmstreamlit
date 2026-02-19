import streamlit as st  # type: ignore
from main.config import Colors
from main.components.image import Image
from streamlit_extras.stylable_container import stylable_container
import streamlit_antd_components as sac

# Función para renderizar la barra lateral donde se incluye el menú de navegación y otros elementos visuales
def sidebar_show():

    # Aplicamos estilos personalizados a la barra lateral
    st.markdown(
        f"""
                    <style>
                        [data-testid="stSidebar"] {{
                            background-color: {Colors.OXYGEN_WHITE};
                        }}
                        [class="st-emotion-cache-1fwbbrh"] {{
                            margin-bottom: 0 !important;
                            text-align: center;
                        }}
                    </style>
                """,
        unsafe_allow_html=True,
    )

    # Creamos el sidebar
    with st.sidebar:
        # Información sobre la libreria: https://extras.streamlit.app/
        with stylable_container(
            key="container_sidebar_home",
            css_styles="""
                {
                    text-align: center;
                }
                label[data-testid="stWidgetLabel"] {
                    text-align: center !important;
                    display: block !important;
                    margin-bottom: 2rem !important;
                }
            """
        ):
            # Mostramos el logo de SEAT y CUPRA
            Image(
                id="logo_seat_cupra",
                url="main/assets/sc_logo_seat_cupra.png",
                local=True,
                width=200,
            ).render()


        st.markdown(f"""
        <style>
            .ant-menu, 
            .ant-menu-sub, 
            .ant-menu-inline, 
            .ant-menu-item, 
            .ant-menu-submenu-title {{
                background-color: transparent !important;
            }}
            .ant-menu-inline, .ant-menu-vertical {{
                border-right: none !important;
            }}
            .ant-menu-item span, .ant-menu-submenu-title span {{
                color: white !important;
            }} 
        </style>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("")

        # Se añade el menú de navegación
        # Información sobre la libreria: https://github.com/nicedouble/StreamlitAntdComponents

        selected = sac.menu([
            sac.MenuItem("MMM Tool", icon='bar-chart-line', children=[
                sac.MenuItem("Sales Drivers", icon='clipboard-data'),
                sac.MenuItem("Marketing ROI", icon='sliders'),
                sac.MenuItem("MM Optimization", icon='rocket-takeoff'),
            ]),
        ],
            open_all=True,
            size='sm',
            color=Colors.CUPRA_COPPER,
            variant='filled'
        )

    # Mapeo: dejamos exactamente los mismos strings que espera page_selected()
    if selected == "Sales Drivers":
        st.session_state["current_page"] = "Sales Drivers"
    elif selected == "Marketing ROI":
        st.session_state["current_page"] = "Marketing ROI"
    elif selected == "MM Optimization":
        st.session_state["current_page"] = "MM Optimization"