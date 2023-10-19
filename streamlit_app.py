# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo lets you to explore the Udacity self-driving car image dataset.
# More info: https://github.com/streamlit/demo-self-driving

"""
https://docs.streamlit.io/library/advanced-features/caching#example-1-pass-a-database-connection-around

Most parameters from st.cache are also present in the new commands, with a few exceptions:

allow_output_mutation does not exist anymore. You can safely delete it. Just make sure you use the right caching command for your use case.

suppress_st_warning does not exist anymore. You can safely delete it. Cached functions can now contain Streamlit commands and will replay them. If you want to use widgets inside cached functions, set experimental_allow_widgets=True. See here.

hash_funcs does not exist anymore. You can exclude parameters from caching (and being hashed) by prepending them with an underscore: _excluded_param. See here.

"""

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2

from pdb import set_trace

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = ""
st.set_page_config(layout="wide", page_title="YOLO v3 オブジェクト検出", page_icon=":taxi:")

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.

    # st.sidebar.title("What to do")

    app_mode = st.sidebar.selectbox("アプリの実行モード",
        ["インストラクションを表示する", "アプリを実行する", "ソース・コードを表示する"])
    if app_mode == "インストラクションを表示する":
        st.sidebar.success('"アプリを実行する" モードを選択するとアプリが走ります。')
    elif app_mode == "ソース・コードを表示する":
        readme_text.empty()
        st.code(get_file_content_as_string("streamlit_app.py"))
    elif app_mode == "アプリを実行する":
        readme_text.empty()
        run_the_app()

# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("ダウンロード中 %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("ダウンロード中 %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.cache_data
    def load_metadata(url):
        return pd.read_csv(url)

    # This function uses some Pandas magic to summarize the metadata Dataframe.
    @st.cache_data
    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
        summary = one_hot_encoded.groupby(["frame"]).sum().rename(columns={
            "label_biker": "自転車", # "biker",
            "label_car": "乗用車", # "car",
            "label_pedestrian": "歩行者", # "pedestrian",
            "label_trafficLight": "信号機", # "traffic light",
            "label_truck": "トラック" # "truck"
        })
        return summary

    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!
    metadata = load_metadata(os.path.join(DATA_URL_ROOT, "labels.csv.gz"))
    summary = create_summary(metadata)

    # Uncomment these lines to peek at these DataFrames.
    # st.write('## Metadata', metadata[:1000], '## Summary', summary[:1000])

    # Draw the UI elements to search for objects (pedestrians, cars, etc.)
    selected_frame_index, selected_frame, object_type = frame_selector_ui(summary)
    if selected_frame_index == None:
        st.error("No frames fit the criteria. Please select different label or number.")
        return

    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold = object_detector_ui()

    # Load the image from S3.
    image_url = os.path.join(DATA_URL_ROOT, selected_frame)
    image = load_image(image_url)

    # Get the boxes for the objects detected by YOLO by running the YOLO model.
    yolo_boxes = yolo_v3(image, confidence_threshold, overlap_threshold)
    draw_image_with_boxes(image, yolo_boxes, "YOLO(v3) オブジェクト検出パラメータ",
        "%s: %i コマ, 重複度 `%3.1f`, 確度 `%3.1f`" % \
            (object_type, selected_frame_index, overlap_threshold, confidence_threshold),
        object_type)

    # Add boxes for objects on the image. These are the boxes for the ground image.
    # boxes = metadata[metadata.frame == selected_frame].drop(columns=["frame"])
    # draw_image_with_boxes(image, boxes, "Ground Truth",
    #     "**Human-annotated data** (frame `%i`)" % selected_frame_index, object_type)


# This sidebar UI is a little search engine to find certain object types.
def frame_selector_ui(summary):

    # st.sidebar.markdown("# Frame")

    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("検出するオブジェクトを指定してください。", summary.columns, 1)

    # The user can select a range for how many of the selected objects should be present.
    min_elts, max_elts = st.sidebar.slider("最大いくつの%sを赤色で表示しますか？" % object_type, 0, 25, [0, 10])
    selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)
    if len(selected_frames) < 1:
        return None, None, object_type

    # Choose a frame out of the selected frames.
    selected_frame_index = st.sidebar.slider(f"動画の{len(selected_frames):,}コマの1つを指定してください。", 0, len(selected_frames) - 1, 1005)

    # Draw an altair chart in the sidebar with information on the frame.
    objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
    chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
        alt.X("index:Q", scale=alt.Scale(nice=False)),
        alt.Y("%s:Q" % object_type))
    selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(x = "selected_frame")
    st.sidebar.altair_chart(alt.layer(chart, vline))

    selected_frame = selected_frames[selected_frame_index]
    return selected_frame_index, selected_frame, object_type

# Select frames based on the selection in the sidebar
@st.cache_resource
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index

# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():

    # st.sidebar.markdown("# Model")

    confidence_threshold = st.sidebar.slider("最低、どの程度の確かさで検出するかを指定してください", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("どの程度重複するオブジェクトまで検出するかを指定してくさい", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
def draw_image_with_boxes(image, boxes, header, description, object_type):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    # LABEL_COLORS = {
    #     "car": [255, 0, 0],
    #     "pedestrian": [0, 255, 0],
    #     "truck": [0, 0, 255],
    #     "trafficLight": [255, 255, 0],
    #     "biker": [255, 0, 255],
    # }
    RED = [255, 0, 0]
    LABEL_COLORS = {
        "乗用車": RED,
        "歩行者": RED,
        "トラック": RED,
        "信号機": RED,
        "自転車": RED,
    }
    image_with_boxes = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        if label == object_type:
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

    # Draw the header and image.
    # st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

# Download a single file and make its content available as a string.
@st.cache_resource(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/Setotet/slit-demo-YOLO/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.cache_data(show_spinner=False)
def load_image(url):
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image

# Run the YOLO model to detect objects.
def yolo_v3(image, confidence_threshold, overlap_threshold):
    # Load the network. Because this is cached it will only happen once.
    @st.cache_resource
    def load_network(config_path, weights_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names
    net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

    # Run the YOLO neural net.
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    # Supress detections in case of too low confidence or too much overlap.
    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)

    # Map from YOLO labels to Udacity labels.
    UDACITY_LABELS = {
        0: '歩行者',
        1: '自転車',
        2: '乗用車',
        3: '自転車',
        5: 'トラック',
        7: 'トラック',
        9: '信号機'
    }
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    if len(indices) > 0:
        # loop over the indexes we are keeping
        for i in indices.flatten():
            label = UDACITY_LABELS.get(class_IDs[i], None)
            if label is None:
                continue

            # extract the bounding box coordinates
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
            labels.append(label)

    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
    return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]

# Path to the Streamlit public S3 bucket
DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": 8342
    }
}

if __name__ == "__main__":
    main()
