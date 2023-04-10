from transformers import pipeline
import streamlit as st
import os
def load_model(text):
    classifier = pipeline("text-classification", model="SkolkovoInstitute/russian_toxicity_classifier")
    return classifier(text)
st.title('Определение токсичности текста')


st.markdown('Проверьте, насколько **токсичен** ваш текст.')
st.markdown('Введите ниже вашу фразу :point_down:')
text = st.text_input('dd', '', placeholder='Введите текст для анализа', label_visibility="hidden")
result = st.button('Задетектить токсика')

if result:
    res = load_model(text)
    st.markdown('Вот такие **результаты** :')
    if res[0]['label'] == 'neutral':
        st.markdown(f'Ваша фраза :green[приемлема] с вероятностью в {round(res[0]["score"],2)}')
    else:
        st.markdown(f'Ваша фраза :red[токсична] с вероятностью в {round(res[0]["score"],2)}')



