
from flask import Flask, render_template, redirect, session, request
from pymongo import MongoClient
import plotly.graph_objects as go
from dash import Dash
import yfinance as yf
import AIML
from dash_bootstrap_components.themes import BOOTSTRAP
from dash.dependencies import Input, Output, State
from dash import dcc, html
import pandas as pd
from datetime import date
import requests, bs4
import otp

app = Flask(__name__)
ticker=[]
app.secret_key = "secret"
MONGO_URL="ENTER MONGODB URL"
DETAIL=[]
stock_ticker=[]

dash_app = Dash(__name__, server=app, url_base_pathname='/dash/', external_stylesheets=[BOOTSTRAP])



@dash_app.callback(
        Output('search-output', 'children'),
        Input('search-button', 'n_clicks'),
        State('search-input', 'value')
)
def update_search_result(n_clicks, search_query):
    ticker=search_query
    print(ticker)
    if n_clicks is not None and n_clicks > 0 and search_query:
        print("tick :",ticker)
        tic=ticker 
        print("route :",tic)
        print(tic)
        dataset=yf.download(tic,start="2010-01-01",end=str(date.today()))
        dataset=pd.DataFrame(dataset)
        print(dataset)
        df,accuracy,mse,rmse,color,value,info=AIML.aiml(tic,dataset)
        print(accuracy,type(accuracy),"\n",mse,type(mse),"\n",rmse,type(rmse))
        time_buttons = [
        {'count': 6, 'step': 'month', 'stepmode': 'todate', 'label': '6M'},
        {'count': 1, 'step': 'year', 'stepmode': 'todate', 'label': '1Y'},
        {'count': 5, 'step': 'year', 'stepmode': 'todate', 'label': '5Y'},
        {'count': 1, 'step': 'all', 'stepmode': 'todate', 'label': 'ALL'}
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dataset.index, y=dataset['Close'], mode='lines', name='Close'))
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(rangeselector={'buttons': time_buttons})
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df["Date"], y=df["Close"],mode="lines",name="Close"))
        fig1.update_layout(hovermode="x unified")
        fig1.update_xaxes(rangeselector={'buttons': time_buttons})
        fig2=go.Figure(go.Candlestick(x=dataset.index, 
                                    open=dataset["Open"],
                                    high=dataset['High'],
                                    low=dataset['Low'],
                                    close=dataset['Close']))
        fig2.update_layout(hovermode="x unified")
        fig2.update_xaxes(rangeselector={'buttons': time_buttons})
        fig2.update_layout(xaxis_rangeslider_visible=False)

        ticker=yf.Ticker(tic)
        data=ticker.get_info()
        return html.Div(className="app-div",
                    children=[
                        html.Div(
                            className="head_id",
                            children=[
                            html.H1(f"{data['longName']} ({data['symbol']})"),
                            html.H2(f"{data['currentPrice']} {data['currency']}")
                        ],style={"font-size": "30px","background-color":"white"}),
                        html.Div(
                            className="graph_id1",
                            children=[
                                dcc.Graph(figure=fig)
                            ]
                        ),
                        html.Div(
                            className="graph_id2",
                            children=[
                                dcc.Graph(figure=fig2)
                            ]
                        ),
                        html.Div(
                            className="head1",
                            children=[
                                html.H2("Overview",style={"margin-bottom":"-10px"}),
                                html.Hr(style={"border-top":"1px solid black"})
                            ],style={"display":"block","postion":"absolute","left":"30px","top":"30pc","background-color":"white"}  
                        ),
                        html.Div(
                            className="details",
                            children=[
                                html.Div(
                                    children=[
                                        html.P("Previous Close"),
                                        html.P(data['previousClose'])
                                    ],
                                    style={"display":"block","padding":30}
                                ),
                                html.Div(
                                    children=[
                                        html.P("Open"),
                                        html.P(data['open'])
                                    ],
                                    style={"display":"block","padding":30}
                                ), 
                                html.Div(
                                    children=[
                                        html.P("Volume"),
                                        html.P(data['volume'])
                                    ],
                                    style={"display":"block","padding":30}
                                ), 
                                html.Div(
                                    children=[
                                        html.P("Average Volume"),
                                        html.P(data['averageVolume'])
                                    ],
                                    style={"display":"block","padding":30}
                                ), 
                                html.Div(
                                    children=[
                                        html.P("High"),
                                        html.P(data['dayHigh'])
                                    ],
                                    style={"display":"block","padding":30}
                                ),
                                html.Div(
                                    children=[
                                        html.P("Low"),
                                        html.P(data['dayLow'])
                                    ],
                                    style={"display":"block","padding":30}
                                ),
                                html.Div(
                                    children=[
                                        html.P("Market Cap"),
                                        html.P(data['marketCap'])
                                    ],
                                    style={"display":"block","padding":30}
                                ), 
                                html.Div(
                                    children=[
                                        html.P("Dividend Rate"),
                                        html.P(data['dividendRate'])
                                    ],
                                    style={"display":"block","padding":30}
                                ),    
                            ],style={"display":"flex","justify-content":"center","font-family":"arial","margin-top":"-20px","background-color":"white"}
                        ),
                        html.Div(
                            className="head1",
                            children=[
                                html.H2(f"About {data['longName']}",style={"margin-bottom":"-10px"}),
                                html.Hr(style={"border-top":"1px solid black"})
                            ],style={"display":"block","postion":"absolute","left":"30px","top":"30pc","background-color":"white"}  
                        ),
                        html.Div(
                            className="details",
                            children=[
                                html.Div(
                                    children=[
                                        html.P(data['longBusinessSummary'])
                                    ],
                                    style={"display":"block","padding":30}
                                ),
                                html.Div(
                                    children=[
                                        html.P("Web Site"),
                                        html.A(data['website'],href=data['website'])
                                    ],style={"display":"block","padding":30,"margin-top":"-60px"}
                                ),
                                html.Div(
                                    children=[
                                        dcc.Graph(figure=fig1)
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.P("Details"),
                                        html.P(children=["MSE : ",mse]),
                                        html.P(children=["RMSE : ",rmse]),
                                        html.P(children=["Accuracy : ",accuracy]),
                                        html.P(children=[f"{info} : ",value],style={"color":color})
                                    ],style={"display":"block","padding":30,"margin-top":"-60px"}
                                )
                            ],style={"display":"block","justify-content":"center","font-family":"arial","margin-top":"-20px","background-color":"white"}
                        ),
                    ],style={"padding":40,"font-family": "sans-serif"})
    else:
        pass
dash_app.layout = html.Div(className="search-dash",
    children=[
    dcc.Input(id='search-input', type='text', placeholder='Enter Stock Ticker',style={"border-radius":"5px","padding":"5px","margin-right":"10px","margin-top":"10px"}),
    html.Button('Search', id='search-button',style={"border-radius":"5px","margin-top":"10px","padding":"5px"}),
    html.Div(id='search-output')
],style={"display":"block","font-family":"arial","text-align":"center","margin":"10px","margin":"10px"})#"background-image":"url('static\signin\images\\bg1.jpg')","background-position":"center","background-repeat":"no-repeat","background-size":"cover"
    
@app.route("/",methods=["GET", "POST"])
def home():
    if request.method == "GET":
        if "user" in session:
            return render_template("/home/index.html",user=session["user"][1])
        return render_template("/home/index.html")
@app.route("/signup",methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        values={"signup":True}
        return render_template("/signin/index.html",values=values)
    elif request.method == "POST":
        user_name = request.form["user_name"]
        user_email =request.form["user_email"]
        passwd = request.form["passwd"]
        client=MongoClient(MONGO_URL)
        info=client["user_info"]
        register_col=info.get_collection("register_details")
        if register_col.find_one({"user_email":user_email,"user_name":user_name}):
            values={"signup":True,"error":"Email already existes"}
            client.close()
            return render_template("/signin/index.html",values=values)
        else:
            register_col.insert_one({"user_email":user_email,"user_name":user_name,"password":passwd})
            client.close()     
            session["user"]=[user_email,user_name]
            return redirect("/")
    return redirect("/")
@app.route("/signin",methods=["GET","POST"])
def signup():
    if request.method == "GET":
        values={"signin":True}
        return render_template("/signin/index.html",values=values)
    elif request.method == "POST":
        user_email = request.form["user_email"]
        passwd = request.form["passwd"]
        client=MongoClient(MONGO_URL)
        info=client["user_info"]
        register_col=info.get_collection("register_details")
        creds=register_col.find_one({"user_email":user_email}) 
        print(creds)
        if creds["user_email"] == user_email  and creds["password"] == passwd:
            session["user"]=[user_email,creds["user_name"]]
            client.close()
            return redirect("/")
        else:
            values={"signin":True,"error":"credentials do not match"}
            client.close()
            return render_template("/signin/index.html",values=values)
    return redirect('/')
@app.route("/reset-password",methods=["GET","POST"])
def reset():
    if request.method == "POST":
        val=request.form
        if "user_email" in val:
            client=MongoClient(MONGO_URL)
            values=client["user_info"]
            print(values)
            register_col=values.get_collection("register_details")
            if register_col.find_one({'user_email':val["user_email"]}):
                print("found")
                DETAIL.clear()
                DETAIL.append(otp.sendotp(val["user_email"]))
                DETAIL.append(val["user_email"])
                client.close()
                print(DETAIL)
                info={"otp":"otp"}
                return render_template("/signin/reset-password.html",info=info)
        elif "otp" in val:
            print(DETAIL)
            info={"rpass":"rpass"}
            if val["otp"] == DETAIL[0]:
                return render_template("/signin/reset-password.html",info=info)
            else:
                info={"error":"In valid OTP","otp":"otp"}
                return render_template("/signin/reset-password.html",info=info)
        elif "passwd" in val:
            print(val["passwd"])
            client=MongoClient(MONGO_URL)["user_info"].get_collection("register_details")
            client.update_one(filter={"user_email":DETAIL[1]},update={"$set":{"password":val["passwd"]}})
            return redirect("/")
    else:
        info={"mail":"email"}
        return render_template("/signin/reset-password.html",info=info)
@app.route("/news",methods=["GET"])
def news():
    news=[]
    r=requests.get("https://www.moneycontrol.com/news/business/stocks/")
    soup=bs4.BeautifulSoup(r.content,"html5lib")
    head=[]
    tail=[]
    other=[]
    for i in range(4):
        d={}
        li_class=soup.find('li',attrs={"id":f"newslist-{i}"})
        d["news_link"]=li_class.find("a").get("href")
        d["news_title"]=li_class.find("a").get("title")
        d["news_image"]=li_class.find("img").get("data").split("?")[0]
        d["news_date"]=li_class.find("span").text
        d["news_discription"]=li_class.find("p").text
        head.append(d)
    for i in range(4,8):
        d={}
        li_class=soup.find('li',attrs={"id":f"newslist-{i}"})
        d["news_link"]=li_class.find("a").get("href")
        d["news_title"]=li_class.find("a").get("title")
        d["news_image"]=li_class.find("img").get("data").split("?")[0]
        d["news_date"]=li_class.find("span").text
        d["news_discription"]=li_class.find("p").text
        tail.append(d)
    for i in range(8,15):
        d={}
        li_class=soup.find('li',attrs={"id":f"newslist-{i}"})
        d["news_link"]=li_class.find("a").get("href")
        d["news_title"]=li_class.find("a").get("title")
        d["news_image"]=li_class.find("img").get("data")
        d["news_date"]=li_class.find("span").text
        d["news_discription"]=li_class.find("p").text
        other.append(d)
    d=date.today()
    dd=date(day=d.day,month=d.month,year=d.year).strftime('%A %d %B %Y')
    return render_template("/news/news.html",head=head,tail=tail,other=other,date=dd)

@app.route('/logout',methods=["GET"])
def logout():
    session.clear()
    return redirect("/")
if __name__ == "__main__":
    app.run(debug=True)