
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image


# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# local_css("style/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
# img_contact_form = Image.open("images/yt_contact_form.png")
# img_lottie_animation = Image.open("images/yt_lottie_animation.png")

# ---- HEADER SECTION ----
with st.container():
    st.header("Welcome To Image Segmenatation for video Software")
    st.title("See the world in a whole new light with video image segmentation!")
    st.write(
        "A user friendly Software"
    )
    st.write("[Learn More >](https://https://paperswithcode.com/task/video-segmentation/)")

# ---- WHAT I DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What It Does")
        # st.write("##")
        st.write(
            """
             Working of the software:\n
            -User Friendly.\n
            -Just A click Away to start analyzing the video.\n
            -Analyze your own video.\n
            \n
            """
        )
        st.write('''<h1>
                 <a target="_self"
                href="{templates/about.html}">To know more </a></h1>''',
                unsafe_allow_html=True)

    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

# ---- PROJECTS ----
with st.container():
    st.write("---")
    st.header("Lets start analyzing the video")
    st.write("##")
   
    fil=0
    but="Open camera"
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(uploaded_file.name)
        fil=uploaded_file.name
        but="run video"
    if st.button(but):
        import cv2
        from visualize_cv2 import model, display_instances, class_names
        import sys

        args = sys.argv
       
	
        stream = cv2.VideoCapture(fil)
	
        while True:
	        ret , frame = stream.read()
	        if not ret:
		        print("unable to fetch frame")
		        break
	        results = model.detect([frame], verbose=1)

	# Visualize results
	        r = results[0]
	        masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
	        cv2.imshow("masked_image",masked_image)
	        if(cv2.waitKey(1) & 0xFF == ord('q')):
		        break
        stream.release()
        cv2.destroyWindow("masked_image")

   