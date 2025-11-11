import dash
from dash import html
import dash_bootstrap_components as dbc

# ======================================================
# 📄 About Us Page Layout
# ======================================================
def about_us_layout():
    return dbc.Container(
        [
            # Top row: Left and Right boxes
            dbc.Row(
                [
                    # Left box (Top Left)
                    dbc.Col(
                        [
                            html.H4("Kalimati Market"),
                            html.P("Kalimaati Market is one of the most important markets in Kathmandu, Nepal, where tomatoes are sold."),
                            
                            # Image Carousel for Kalimati Market
                            dbc.Carousel(
                                items=[
                                    {"key": 1, "src": "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/2a/ce/96/9f/caption.jpg?h=500&s=1&w=900", "header": "Market View 1", "caption": "Early morning at Kalimati"},
                                    {"key": 2, "src": "https://assets-cdn.kathmandupost.com/uploads/source/news/2021/money/thumb2-1620786103.jpg", "header": "Market View 2", "caption": "Tomato stalls galore"},
                                    {"key": 3, "src": "https://assets-cdn.kathmandupost.com/uploads/source/news/2018/others/kalimati-tarkari-bazar01-6-2082018051306-1000x0-2582018060710-1000x01-0492018043716-1000x0-copy_1537323568.jpg", "header": "Market View 3", "caption": "Wholesale buyers at work"},
                                    {"key": 4, "src": "https://republicaimg.nagariknewscdn.com/shared/web/uploads/media/Kalimati-fruits-and-vegetables-market.jpg", "header": "Market View 4", "caption": "Kalimati market bustling with activity"}
                                ],
                                controls=True,  # Enable carousel controls (previous/next buttons)
                                indicators=True,  # Enable indicators (dots below the carousel)
                                interval=2000,  # Set auto-slide interval (2 seconds)
                            )
                        ], 
                        width=6,
                        style={"height": "400px", "border": "1px solid #ccc", "padding": "20px", "overflow": "auto"}
                    ),
                    # Right box (Top Right)
                    dbc.Col(
                        [
                            html.H4("🍅 Tomato Farming"),
                            html.P("This section explains the farming techniques for tomatoes in the region."),
                            
                            # Tomato farming images
                            html.Img(src="https://englishassets.deshsanchar.com/wp-content/uploads/2023/06/18132237/Tomato_kalimati-1-1536x1024-1.jpg", style={"width": "100%", "height": "80%", "object-fit": "cover"}),  # Tomato image 1
                            html.Br(),
                            html.Img(src="https://english.onlinekhabar.com/wp-content/uploads/2017/01/isard2.jpg", style={"width": "100%", "height": "80%", "object-fit": "cover"}),  # Tomato image 2
                        ], 
                        width=6,
                        style={"height": "400px", "border": "1px solid #ccc", "padding": "20px", "overflow": "auto"}
                    ),
                ]
            ),

            # Bottom row: Left and Right boxes for Data Collection (Sudoku-like grid)
            dbc.Row(
                [
                    # Data Collection Section
                    dbc.Col(
                        [
                            html.H4("🔬 Data Collection Methods", style={"textAlign": "center", "color": "#0044cc"}),

                            # Explanation of Data Collection Methods
                            html.P("""
                            We gather data from various trusted sources to ensure accurate and reliable forecasting of tomato prices. 
                            The data we use comes from the following key sources:
                            """),
                            html.Ul([
                                html.Li("📊 **Vegetable Price Data from Kalimati Market Website**: We collect real-time vegetable price data, including tomato prices, from the Kalimati Market website. This helps us track local price trends and understand market fluctuations."),
                                html.Li("🌦 **Weather Data from Open Meteo**: We rely on Open Meteo for up-to-date and historical weather data, including temperature, humidity, and rainfall, which play a crucial role in tomato growth and yield."),
                                html.Li("📈 **Inflation and Exchange Rate Data from Nepal Rastra Bank**: We use data from Nepal Rastra Bank to track inflation rates and exchange rates, which significantly affect the price of imported goods and domestic agricultural products like tomatoes."),
                                html.Li("⛽ **Fuel Data from Nepal Oil Corporation**: Fuel prices are monitored through Nepal Oil Corporation, as changes in fuel prices can impact transportation costs, affecting overall tomato pricing.")
                            ]),
                            html.P("""
                            By integrating these data sources, we can build a more robust and accurate model for forecasting tomato prices in the region, 
                            taking both local market factors and external influences into account.
                            """)
                        ], 
                        width=6,
                        style={"height": "400px", "border": "1px solid #ccc", "padding": "20px", "overflow": "auto"}
                    ),

                    # Model Deployment Section
                    dbc.Col(
                        [
                            html.H4("📊 Model Deployment"),
                            html.P("""
                            The trained machine learning models for tomato price forecasting are deployed on an AWS EC2 (Elastic Compute Cloud) instance. 
                            Once the models are deployed, they are hosted on the EC2 server and made accessible via this web dashboard for real-time price forecasting.
                            """),
                            html.P("""
                            Here’s how the deployment process works:
                            1. **AWS EC2 Instance**: The trained models are hosted on an EC2 instance, a scalable cloud server provided by Amazon Web Services (AWS). This ensures that the models can handle various loads and perform forecasting tasks efficiently.
                            2. **Model Access**: Once deployed, the models are accessible from the dashboard interface, where users can input specific data (such as weather conditions, inflation rates, etc.) and receive real-time tomato price forecasts.
                            3. **Scalability and Reliability**: AWS EC2 provides the scalability to run the models even during periods of high demand and ensures the reliability of the system with its global infrastructure.
                            4. **Web Dashboard Integration**: The web dashboard allows users to interact with the models easily, enter the necessary inputs, and get predictions based on the most up-to-date data.
                            """),
                            html.P("""
                            The deployment on AWS ensures that the forecasting process is automated, secure, and easily accessible for stakeholders. 
                            The real-time predictions can then be used for decision-making in areas like supply chain management, pricing strategies, and market analysis.
                            """),
                            html.Hr(),
                            html.Div([ 
                                html.P("Follow us on: "),
                                html.A("GitHub", href="https://github.com/your_github_link", target="_blank", className="mr-3"),
                                html.A("LinkedIn", href="https://linkedin.com", target="_blank", className="mr-3"),
                                html.A("Facebook", href="https://facebook.com", target="_blank"),
                            ], className="text-center")
                        ], 
                        width=6,
                        style={"height": "400px", "border": "1px solid #ccc", "padding": "20px", "overflow": "auto"}
                    ),
                ]
            ),
        ],
        fluid=True,
    )
