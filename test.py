import streamlit as st

# Function to display page content based on selection
def page_home():
    st.subheader('Home Page')
    st.write('Welcome to the Home Page!')

def page_about():
    st.subheader('About Page')
    st.write('Welcome to the About Page! This is where you can learn more about us.')

def page_contact():
    st.subheader('Contact Page')
    st.write('Welcome to the Contact Page. Feel free to reach out to us.')

def page_help():
    st.subheader('Help Page')
    st.write('Welcome to the Help Page. Here, we provide assistance and support.')



st.sidebar.title('Navigation')
pages = ['Home', 'About', 'Contact', 'Help']
selection = st.sidebar.radio('Go to', pages)

if selection == 'Home':
    page_home()
elif selection == 'About':
    page_about()
elif selection == 'Contact':
    page_contact()
elif selection == 'Help':
    page_help()

