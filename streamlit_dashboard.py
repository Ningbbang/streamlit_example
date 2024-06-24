import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import warnings

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Wokrhour', 'Attrition', 'Map', 'Text Analysis', 'Clustering'])

with tab1:
    st.write('근무시간 분석')

    col1, col2 = st.columns(2)

    st.sidebar.write('근무시간 분석용')
    target_month = st.sidebar.selectbox('날짜 검색', ('all', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'))
    target_dept = st.sidebar.selectbox('부서 검색', ('all', 'A', 'B', 'C', 'D'))

    df = pd.read_csv('https://raw.githubusercontent.com/Ningbbang/personal_project/main/workhour.csv', index_col=0)

    # 근무시간 한도

    if target_month != 'all':
        if target_dept != 'all':
            target_df = df.loc[ (df['Department']==target_dept) & (df['Month']==int(target_month)), :]
        else:
            target_df = df.loc[df['Month']==int(target_month), :]

    elif target_month == 'all':
        if target_dept != 'all':
            target_df = df.loc[df['Department']==target_dept, :]
        else:
            target_df = df.loc[:,:]

    target_df['ot_percent'] = np.round(target_df['net_ot'],1)
    target_df['ot_percent_rev'] = np.round(40 - target_df['net_ot'],1)

    labels = ['순수연장', '잔여시간']
    values = [target_df.iloc[0, -2], target_df.iloc[0, -1]]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, legend=None)], layout=go.Layout(title='연장근무시간'))
    fig.update_layout(
        legend_entrywidth=100,
        legend_yanchor="top",
        legend_y=0.99,
        legend_xanchor="left",
        legend_x=0.01
    )
    col1.write(fig)

    # 휴일비중
    labels = ['휴일근무', '순수연장']
    values = [np.round(target_df['weekend_ot'].mean(),1), np.round(target_df['net_ot'].mean(),1)]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, legend=None)], layout=go.Layout(title='휴일근무비중'))
    fig.update_layout(
        legend_entrywidth=100,
        legend_yanchor="top",
        legend_y=0.99,
        legend_xanchor="left",
        legend_x=0.01
    )
    col2.write(fig)

    fig = px.strip(target_df, x='rank', y='net_ot', hover_name='Emp No', hover_data='net_ot', stripmode='overlay', title='Distribution')
    fig.update_xaxes(categoryorder='array', categoryarray= ['G1S', 'G1D', 'G2', 'G3'])
    col1.write(fig)

    fig = px.box(target_df, x='rank', y='net_ot', title='Boxplot')
    fig.update_xaxes(categoryorder='array', categoryarray= ['G1S', 'G1D', 'G2', 'G3'])
    col2.write(fig)

    df_group = df.groupby(['Department', 'Month']).mean(['net_ot'])
    df_group = df_group.reset_index()
    df_group.pivot_table(index='Department', columns='Month', values='net_ot')

    fig = px.line(df_group, x='Month', y='net_ot', color='Department', title='23년 부서별 평균 연장근무시간')
    st.write(fig)

with tab2:
    df = pd.read_csv('https://raw.githubusercontent.com/Ningbbang/personal_project/main/hr_data.csv')

    left_column, right_column, far_right = st.columns([2, 5, 3])

    left_column.write('퇴직자 예측')
    left_column.dataframe(df)

    test_size = left_column.slider('테스트 사이즈', 0.01, 0.99, (0.2))
    validate_button = left_column.button('검증')


    # Preprocess the dataframe
    @st.cache_data
    def preprocessing(df):
        df.loc[df['Attrition'] == 'Yes', 'Attrition'] = 1
        df.loc[df['Attrition'] == 'No', 'Attrition'] = 0

        y_target = df[['Attrition']]
        y_target = y_target.astype('int64')
        y_target = np.array(y_target).reshape(-1, )
        X_feature = df.drop(columns=['Employee Number', 'Employee Count', 'CF_current Employee',
                                     'emp no', 'CF_attrition label', 'Over18', 'Attrition'])

        object_dtypes_col = X_feature.select_dtypes('object').columns.drop(['Gender', 'Over Time'])
        int_dtypes_col = X_feature.select_dtypes('int64').columns

        for col in object_dtypes_col:
            X_feature = pd.get_dummies(X_feature, columns=[col])

        ss = StandardScaler()
        le = LabelEncoder()

        for col in int_dtypes_col:
            X_feature[col] = ss.fit_transform(X_feature[[col]])

        X_feature['Gender'] = le.fit_transform(X_feature['Gender'])
        X_feature['Over Time'] = le.fit_transform(X_feature['Over Time'])

        return X_feature, y_target


    # Show bar chart of feature importances
    def show_fi(X, y, test_size=0.2, top_show=20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=11)
        rf_clf = RandomForestClassifier(random_state=11, n_estimators=100)
        rf_clf.fit(X_train, y_train)

        if top_show == -1:
            top_show = ''

        fi = pd.Series(data=np.round(rf_clf.feature_importances_, 4), index=X_train.columns).sort_values(
            ascending=False)[
             :20]
        fig = px.bar(fi, x=fi.values, y=fi.index, color=np.round(fi.values, 4),
                     hover_data=[np.round(fi.values, 4)], labels={'x': 'Importance', 'index': 'Feature'})

        fig.update_layout(yaxis=dict(autorange="reversed"))

        return fig


    # Show correlation heatmap
    def show_corr(df):
        X = df.copy()

        X.loc[X['Attrition'] == 'Yes', 'Attrition'] = 1
        X.loc[X['Attrition'] == 'No', 'Attrition'] = 0

        X['Attrition'] = X['Attrition'].astype('int64')

        corr = X.select_dtypes('int64').corr()
        fig = px.imshow(corr, text_auto=False, width=800, height=800)
        return fig


    X_feature, y_target = preprocessing(df)
    left_column.write('예측정확도')

    if validate_button:
        X_feature, y_target = preprocessing(df)
        X_train, X_test, y_train, y_test = train_test_split(X_feature, y_target, test_size=test_size, random_state=11)
        rf_clf = RandomForestClassifier(random_state=11, n_estimators=100)
        rf_clf.fit(X_train, y_train)

        y_pred = rf_clf.predict(X_test)

        accuracy = np.round(accuracy_score(y_test, y_pred), 4)
        precision = np.round(precision_score(y_test, y_pred), 4)
        recall = np.round(recall_score(y_test, y_pred), 4)
        f1 = np.round(f1_score(y_test, y_pred), 4)

        left_column.write(f'Accuracy Score : {accuracy}')
        left_column.write(f'Precision Score : {precision}')
        left_column.write(f'Recall Score : {recall}')
        left_column.write(f'F1 Score : {f1}')

    fig = show_fi(X_feature, y_target, test_size)
    right_column.write('Feature 중요도')
    right_column.plotly_chart(fig, use_container_width=True)

    fig = show_corr(df)
    right_column.write('Feature간 상관관계')
    right_column.plotly_chart(fig, use_container_width=True)

    far_right.write('테스트 데이터 첨부')
    upload_file = far_right.file_uploader('파일을 첨부 해주세요.', type=['csv'])
    predict_button = far_right.button('예측')

    if predict_button:
        if upload_file is None:
            X_feature, y_target = preprocessing(df)
            X_train, X_test, y_train, y_test = train_test_split(X_feature, y_target, test_size=test_size,
                                                                random_state=11, stratify=y_target)
            rf_clf = RandomForestClassifier(random_state=11, n_estimators=100)
            rf_clf.fit(X_train, y_train)

            test_df = pd.read_csv('https://raw.githubusercontent.com/Ningbbang/personal_project/main/test_hr_data.csv')
            X_feature, y_target = preprocessing(test_df)

            missing_columns_from_test_data = X_train.columns.difference(X_feature.columns).tolist()
            for col in missing_columns_from_test_data:
                X_feature[col] = 0
            missing_columns_from_train_data = X_feature.columns.difference(X_train.columns).tolist()
            if len(missing_columns_from_train_data) > 0:
                X_feature.drop(columns=[missing_columns_from_train_data], axis=1, inplace=True)

            y_pred = rf_clf.predict(X_feature)

            result = pd.DataFrame(pd.concat([test_df['emp no'], pd.Series(y_pred)], axis=1))
            result.columns = ['emp no', 'Attrition']

            far_right.write(f'예측 인원수 : {result.shape[0]}')
            predicted = result.loc[result['Attrition'] == 1, :].shape[0]
            far_right.write(f'예측 퇴사자 : {predicted}')
            far_right.write('예측 결과')
            far_right.dataframe(result.loc[result['Attrition'] == 1, :])
            far_right.write('최초 테스트 데이터')
            far_right.dataframe(test_df)

with tab3:
    import folium
    import json
    import requests
    from streamlit_folium import folium_static
    from folium.plugins import HeatMap
    from branca.colormap import StepColormap

    col1, col2 = st.columns(2)
    df = pd.read_csv('https://raw.githubusercontent.com/Ningbbang/personal_project/main/address_data.csv', encoding='utf8')
    url = 'https://raw.githubusercontent.com/Ningbbang/personal_project/main/skorea_municipalities_geo_simple.json'
    resp = requests.get(url)
    geo_data=json.loads(resp.text)

    col1.write('Map')
    df_group = df.groupby(['district']).count()[['address']]
    df_group.reset_index(inplace=True)
    bonsa_coord = (37.1251679, 127.0826964)
    kihueng_coord = (37.21929069, 127.1076686)

    # marker
    m = folium.Map(location=[37.18, 127.0956], zoom_start=12)
    bonsa = folium.Marker(location=bonsa_coord, popup='본사').add_to(m)
    kihueng = folium.Marker(location=kihueng_coord, popup='기흥').add_to(m)
    with col1:
        folium_static(m)

    # choropleth
    col2.write('Choropleth')
    choropleth = folium.Choropleth(
        geo_data=geo_data,
        data=df_group,
        columns=['district', 'address'],
        legend_name='직원수',
        key_on='feature.properties.name'
    ).add_to(m)
    with col2:
        folium_static(m)


    # marker cluster
    col1.write('Marker Cluster')
    from folium import plugins

    m = folium.Map([37.18, 127.09], zoom_start=12)
    bonsa = folium.Marker(location=bonsa_coord, popup='본사').add_to(m)
    kihueng = folium.Marker(location=kihueng_coord, popup='기흥').add_to(m)
    location = df[['lat', 'lng']]
    plugins.MarkerCluster(location).add_to(m)
    with col1:
        folium_static(m)

    # circle marker
    col2.write('Circle Marker')
    m = folium.Map([37.18, 127.09], zoom_start=12)
    bonsa = folium.Marker(location=bonsa_coord, popup='본사').add_to(m)
    kihueng = folium.Marker(location=kihueng_coord, popup='기흥').add_to(m)
    locations = list(zip(df.lat, df.lng))
    for i in range(len(location)):
        folium.CircleMarker(location=locations[i], radius=2, color='red').add_to(m)
    with col2:
        folium_static(m)

with tab4:
    from wordcloud import WordCloud, STOPWORDS
    from konlpy.tag import Okt
    from PIL import Image
    import requests
    from bs4 import BeautifulSoup
    import numpy as np
    import re
    import matplotlib.pyplot as plt
    from io import BytesIO

    col1, col2 = st.columns(2)
    col1.write('원익IPS 텍스트 분석')
    col2.write('콰이어트 플레이스:첫번째날 리뷰 분석')

    url = 'https://raw.githubusercontent.com/Ningbbang/personal_project/main/test_text.txt'
    txt = requests.get(url).text
    txt = re.sub('\n', '', txt)
    t = Okt()
    tokens_ko = t.nouns(txt)
    token = []

    for i, v in enumerate(tokens_ko):
        if len(v) > 1:
            token.append(v)

    token_series = pd.Series(data=token, index=range(len(token)))
    freq_df = pd.DataFrame(token_series.value_counts()).reset_index(drop=False)
    freq_df.columns = ['word', 'count']

    word = {}
    for idx, row in freq_df.iterrows():
        word[row['word']] = row['count']

    url = 'https://raw.githubusercontent.com/Ningbbang/personal_project/main/quiet_place_review_data.txt'
    reviews_text = requests.get(url).text

    #url = 'https://www.imdb.com/title/tt6644200/reviews'
    #response = requests.get(url)
    #soup = BeautifulSoup(response.text, 'html.parser')
    #reviews = soup.find_all('div', 'text show-more__control')
    #reviews_text = ''

    #for review in reviews:
    #    reviews_text += review.text

    clean = re.compile(r'[^a-zA-Z0-9\s]+')
    reviews_text = re.sub(clean, '', reviews_text)

    wordcloud = WordCloud(
        font_path='https://github.com/Ningbbang/personal_project/blob/main/NanumGothic.ttf',
        relative_scaling=0.2,
        background_color='white'
    ).generate_from_frequencies(word)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(wordcloud)
    ax.axis('off')
    col1.pyplot(fig)

    stopwords = set(STOPWORDS)
    stopwords.add('movie')
    stopwords.add('film')
    stopwords.add('one')
    stopwords.add('make')

    wordcloud = WordCloud(
        background_color='white', max_words=1000, stopwords=stopwords)
    wordcloud = wordcloud.generate(reviews_text)
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    col2.pyplot(fig)

    url = 'https://raw.githubusercontent.com/Ningbbang/personal_project/main/IPS_logo.png'
    response = requests.get(url)
    logo = Image.open(BytesIO(response.content))
    logo = np.array(logo)
    # wordcloud

    wordcloud = WordCloud(
        font_path='https://github.com/Ningbbang/personal_project/blob/main/NanumGothic.ttf',
        relative_scaling=0.2,
        mask=logo,
        background_color='white'
    ).generate_from_frequencies(word)

    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    col1.pyplot(fig)

    wordcloud = WordCloud(
        font_path='https://github.com/Ningbbang/personal_project/blob/main/NanumGothic.ttf',
        relative_scaling=0.2,
        mask=logo,
        background_color='white',
        stopwords=stopwords
    ).generate(reviews_text)

    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    col2.pyplot(fig)

    # 최빈 30개 단어
    import plotly.express as px
    fig = px.line(freq_df.iloc[:30, :], x='word', y='count', title='최빈 30개 단어')
    col1.write(fig)

with tab5:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()

    n_cluster = int(st.number_input('클러스터 개수 입력', min_value=2, max_value=8, value=4))
    col1, col2 = st.columns(2)
    df = pd.read_csv('https://raw.githubusercontent.com/Ningbbang/personal_project/main/workhour.csv', index_col=0)
    df_pivot = pd.pivot_table(data=df, index='Emp No',
                              values=['week_ot', 'weekend_ot', 'net_ot', 'weekend_ratio'], aggfunc=['mean'])
    df_pivot.columns = ['week_ot', 'weekend_ot', 'net_ot', 'weekend_ratio']
    df_pivot_std = pd.DataFrame(data=ss.fit_transform(df_pivot),
                                columns=['week_ot', 'weekend_ot', 'net_ot', 'weekend_ratio'])
    kmean = KMeans(n_clusters=n_cluster)
    kmean.fit(df_pivot[['net_ot', 'weekend_ratio']])
    df_pivot['cluster'] = kmean.predict(df_pivot[['net_ot', 'weekend_ratio']])
    df_pivot_std['cluster'] = kmean.fit_predict(df_pivot_std[['net_ot', 'weekend_ratio']])
    df_pivot['cluster'] = df_pivot_std['cluster']
    fig = px.scatter(df_pivot_std, x='week_ot', y='weekend_ratio', color='cluster')

    col1.write('순수연장, 휴일근무비로 클러스터링')
    col1.dataframe(df_pivot)
    col2.write(fig)

    col1.write('설문결과 예측하기 - 학습 데이터')

    result = pd.read_csv('https://raw.githubusercontent.com/Ningbbang/personal_project/main/survey_result.csv',
                         index_col=0)
    result.drop(['Month'], axis=1, inplace=True)
    col1.dataframe(result)

    result_ohe = pd.get_dummies(result, columns=['Department', 'rank'])
    from sklearn.linear_model import LinearRegression

    question_list = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']

    result_test = pd.read_csv('https://raw.githubusercontent.com/Ningbbang/personal_project/main/survey_test.csv',
                              index_col=0)
    X_test = pd.get_dummies(result_test, columns=['Department', 'rank'])
    X_test = X_test.drop(question_list, axis=1)
    X_test = X_test.drop(columns=['Emp No'], axis=1)
    X_feature = result_ohe.drop(columns=question_list, axis=1)
    X_feature = X_feature.drop(columns=['Emp No'], axis=1)
    result = {}

    for i, question in enumerate(question_list):
        y_target = result_ohe[question]

        lr = LinearRegression()
        lr.fit(X_feature, y_target)
        result[question] = lr.predict(X_test)

    result_df = pd.DataFrame(result)
    result_df = pd.concat([result_test, result_df], axis=1)
    result_by_dept = pd.pivot_table(result_df, index='Department', values=question_list, aggfunc=['mean'])
    result_by_dept.columns = question_list
    col2.write('예측 데이터 부서별 평균')
    col2.dataframe(result_by_dept)

    result_by_rank = pd.pivot_table(result_df, index='rank', values=question_list, aggfunc=['mean'])
    result_by_rank.columns = question_list
    col2.write('예측 데이터 직급별 평균')
    col2.dataframe(result_by_rank)

    address = pd.read_csv('https://raw.githubusercontent.com/Ningbbang/personal_project/main/address_data.csv',
                          index_col=0)
    address['lat_std'] = ss.fit_transform(address[['lat']])
    address['lng_std'] = ss.fit_transform(address[['lng']])
    kmeans = KMeans(n_clusters=5)
    address['cluster'] = kmeans.fit_predict(address[['lat', 'lng']])

    m = folium.Map(location=[37.18, 127.0956], zoom_start=11, width='100%', height='100%')
    centers = list(zip(kmeans.cluster_centers_.T.reshape(2, 5)[0], kmeans.cluster_centers_.T.reshape(2, 5)[1]))
    colors = ['red', 'blue', 'green', 'purple', 'black']
    radius = [50, 50, 50, 50, 50]

    st.write('거주지별 클러스터(n=5)')
    for i, color in enumerate(colors):
        locations = list(zip(address.loc[address['cluster'] == i, 'lat'], address.loc[address['cluster'] == i, 'lng']))
        folium.Circle(location=centers[i], radius=radius[i] * 50, color=color, fill_color='olive',
                      fill_opacity=0.2).add_to(m)
        for j in range(len(locations)):
            folium.CircleMarker(location=locations[j], radius=2, color=color).add_to(m)
    folium_static(m, width=720, height=600)