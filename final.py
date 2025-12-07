#final project linear algebra: Dyastiti-Group 7-Final Report
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Matrix Image Lab — 2D Transforms & Filters", layout="wide")

# Matrix helpers (your functions adapted)
def translation(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=np.float64)
def scaling(sx, sy, center=(0,0)):
    cx, cy = center
    T1 = translation(-cx, -cy)
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0, 0, 1]], dtype=np.float64)
    T2 = translation(cx, cy)
    return T2 @ S @ T1
def rotation(angle_deg, center=(0,0)):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    cx, cy = center
    T1 = translation(-cx, -cy)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float64)
    T2 = translation(cx, cy)
    return T2 @ R @ T1
def shear(shx, shy, center=(0,0)):
    cx, cy = center
    T1 = translation(-cx, -cy)
    Sh = np.array([[1, shx, 0],
                   [shy, 1, 0],
                   [0,   0, 1]], dtype=np.float64)
    T2 = translation(cx, cy)
    return T2 @ Sh @ T1
def reflection(axis, axis_pos=None, img_shape=None):
    if img_shape is not None and axis_pos is None:
        h, w = img_shape[:2]
        axis_pos = w/2 if axis=='vertical' else h/2
    if axis == 'vertical':
        T1 = translation(-axis_pos, 0)
        R = np.array([[-1,0,0],
                      [0,1,0],
                      [0,0,1]], dtype=np.float64)
        T2 = translation(axis_pos, 0)
        return T2 @ R @ T1
    else:
        T1 = translation(0, -axis_pos)
        R = np.array([[1,0,0],
                      [0,-1,0],
                      [0,0,1]], dtype=np.float64)
        T2 = translation(0, axis_pos)
        return T2 @ R @ T1

# Image IO helpers
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    arr = np.array(img) 
    return arr

def pil_from_cv(img_cv):
    return Image.fromarray(np.clip(img_cv,0,255).astype(np.uint8))

# Apply homography to image (use computed 3x3)
def apply_homography_to_image(img_rgb, H, dst_size=None, border_mode=cv2.BORDER_CONSTANT):
    """
    img_rgb: HxWx3 numpy array in RGB
    H: 3x3 homography (numpy)
    dst_size: (w, h) tuple. If None, keep same
    returns transformed image (RGB)
    """
    h, w = img_rgb.shape[:2]
    if dst_size is None:
        dst_w, dst_h = w, h
    else:
        dst_w, dst_h = dst_size
    H_cv = H.astype(np.float64)
    transformed = cv2.warpPerspective(img_rgb, H_cv, (dst_w, dst_h),
                                      flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(255,255,255))
    return transformed
def manual_convolve(img, kernel):
    """
    img: HxW or HxWx3 (uint8 or float)
    kernel: kxk float
    returns image same shape float32 clipped to [0,255]
    """
    if img.ndim == 2:
        channels = 1
        h, w = img.shape
        img_pad = np.pad(img.astype(np.float32), ((kernel.shape[0]//2,)*2, (kernel.shape[0]//2,)*2), mode='reflect')
        out = np.zeros((h, w), dtype=np.float32)
        k = kernel.shape[0]
        for i in range(h):
            for j in range(w):
                patch = img_pad[i:i+k, j:j+k]
                out[i,j] = np.sum(patch * kernel)
        return np.clip(out, 0, 255)
    else:
        h, w, c = img.shape
        out = np.zeros_like(img, dtype=np.float32)
        for ch in range(c):
            out[:,:,ch] = manual_convolve(img[:,:,ch], kernel)
        return np.clip(out,0,255)

def blur_kernel(size=3):
    k = np.ones((size,size), dtype=np.float32)
    k = k / k.sum()
    return k
def sharpen_kernel():
    return np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)

def remove_background_hsv(img_rgb, lower_hsv, upper_hsv):
    """Keep pixels inside HSV range as foreground; returns mask and result image with white background"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask_bool = mask>0
    out = img_rgb.copy()
    out[~mask_bool] = 260
    return mask, out

def grabcut_bkg_removal(img_rgb, rect=None, iter_count=5):
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h,w), np.uint8)
    if rect is None:
        rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    out = img_rgb * mask2[:,:,None] + 255*(1-mask2)[:,:,None]
    return mask2*255, out.astype(np.uint8)

# UI: Multi-page
pages = ["Home / Introduction", "Image Processing Tools", "Team Members"]
page = st.sidebar.radio("Pages", pages)

# Common small helper to display two images
def show_side_by_side(orig, transformed, w=350):
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_from_cv(orig), caption="Original", use_column_width=True)
    with col2:
        st.image(pil_from_cv(transformed), caption="Transformed / Filtered", use_column_width=True)

# Page: Home
if page == "Home / Introduction":
    st.title("Matrix Image Lab — 2D Homogeneous Transforms & Convolution Filters")
    st.markdown("""
    **What this app does**
    - Apply 2D transformations (Translation, Scaling, Rotation, Shearing, Reflection) using 3×3 homogeneous matrices.
    - Apply convolution-based image filters (Blur, Sharpen) using custom kernels implemented manually.
    - Optional: simple background removal (HSV threshold or GrabCut).
    - Multi-page UI: this intro, Image Processing Tools, Team Members.

    **Notes**
    - Transformations are constructed manually as 3×3 matrices and then applied to the image using a homography (warpPerspective).
    - Blur and Sharpen are implemented with explicit convolution loops (no OpenCV blur).
    - Larger images may be slow for manual convolution; downscale for experimentation then export at full resolution if needed.
    """)
    st.markdown("**Visual examples**")
    demo_img = np.zeros((220,220,3), dtype=np.uint8)
    demo_img[60:160,40:180] = [200,60,60]
    M_demo = rotation(30, center=(110,110)) @ scaling(0.7, 0.7, center=(110,110)) @ translation(20,-10)
    demo_trans = apply_homography_to_image(demo_img, M_demo, dst_size=(220,220))
    colA, colB = st.columns(2)
    with colA:
        st.image(pil_from_cv(demo_img), caption="Original (demo)", use_column_width=True)
    with colB:
        st.image(pil_from_cv(demo_trans), caption="Rot+Scale+Trans (demo)", use_column_width=True)
    st.stop()

# Page: Image Processing Tools
elif page == "Image Processing Tools":
    
    st.title("Image Processing Tools")
    st.markdown("Upload an image, select transform(s) and/or filter(s), preview and download the result.")
    
    uploaded = st.file_uploader("Upload an image (jpg/png). Large images may be slow.", type=["jpg","jpeg","png"])
    if uploaded is None:
        st.info("Upload an image to begin. Meanwhile you can read the left sidebar to prepare parameters.")
        st.stop()
    img = load_image(uploaded) 
    st.image(uploaded)

    st.sidebar.header("Image Transform Controls")
    transform_mode = st.sidebar.radio("Transformation mode", ["Single transform", "Composite builder"])
    if transform_mode == "Single transform":
        tchoice = st.sidebar.selectbox("Transform", ["Translation","Scaling","Rotation","Shearing","Reflection","None"])
        H = np.eye(3, dtype=np.float64)
        h, w = img.shape[:2]
        center = (w/2, h/2)
        if tchoice == "Translation":
            tx = st.sidebar.number_input("tx (px)", value=0.0, format="%.1f", key="t_tx")
            ty = st.sidebar.number_input("ty (px)", value=0.0, format="%.1f", key="t_ty")
            H = translation(tx, ty)
        elif tchoice == "Scaling":
            sx = st.sidebar.number_input("sx", value=1.0, format="%.3f", key="t_sx")
            sy = st.sidebar.number_input("sy", value=1.0, format="%.3f", key="t_sy")
            H = scaling(sx, sy, center=center)
        elif tchoice == "Rotation":
            ang = st.sidebar.slider("Angle (deg)", -180.0, 180.0, 0.0, key="t_ang")
            H = rotation(ang, center=center)
        elif tchoice == "Shearing":
            shx = st.sidebar.number_input("shx", value=0.0, format="%.3f", key="t_shx")
            shy = st.sidebar.number_input("shy", value=0.0, format="%.3f", key="t_shy")
            H = shear(shx, shy, center=center)
        elif tchoice == "Reflection":
            axis = st.sidebar.radio("Axis", ["vertical","horizontal"], key="t_ref_axis")
            pos = None
            H = reflection(axis, axis_pos=pos, img_shape=img.shape)
        else:
            H = np.eye(3)
    else:
        st.sidebar.markdown("Build composite (left-multiply to apply new transform after existing).")
        if "composite_H" not in st.session_state:
            st.session_state.composite_H = np.eye(3, dtype=np.float64)
        ch = st.sidebar.selectbox("Add transform:", ["Translation","Scaling","Rotation","Shearing","Reflection"])
        cx, cy = img.shape[1]/2, img.shape[0]/2
        if ch == "Translation":
            tx = st.sidebar.number_input("tx (px)", value=0.0, key="c_tx")
            ty = st.sidebar.number_input("ty (px)", value=0.0, key="c_ty")
            M_new = translation(tx,ty)
        elif ch == "Scaling":
            sx = st.sidebar.number_input("sx", value=1.0, key="c_sx")
            sy = st.sidebar.number_input("sy", value=1.0, key="c_sy")
            M_new = scaling(sx, sy, center=(cx,cy))
        elif ch == "Rotation":
            ang = st.sidebar.slider("Angle (deg)", -180.0, 180.0, 0.0, key="c_ang")
            M_new = rotation(ang, center=(cx,cy))
        elif ch == "Shearing":
            shx = st.sidebar.number_input("shx", value=0.0, key="c_shx")
            shy = st.sidebar.number_input("shy", value=0.0, key="c_shy")
            M_new = shear(shx, shy, center=(cx,cy))
        else:
            axis = st.sidebar.radio("Axis", ["vertical","horizontal"], key="c_ref_axis")
            M_new = reflection(axis, img_shape=img.shape)
        if st.sidebar.button("Left-multiply add"):
            st.session_state.composite_H = M_new @ st.session_state.composite_H
        if st.sidebar.button("Reset composite"):
            st.session_state.composite_H = np.eye(3, dtype=np.float64)
        st.sidebar.markdown("Current composite matrix:")
        st.sidebar.write(st.session_state.composite_H)
        H = st.session_state.composite_H
    # target canvas size controls
    st.sidebar.header("Output canvas")
    keep_size = st.sidebar.checkbox("Keep original size", value=True)
    if keep_size:
        dst_size = (img.shape[1], img.shape[0])
    else:
        out_w = st.sidebar.number_input("out width (px)", value=img.shape[1], step=1)
        out_h = st.sidebar.number_input("out height (px)", value=img.shape[0], step=1)
        dst_size = (int(out_w), int(out_h))
    # Filters
    st.sidebar.header("Filters (convolution)")
    apply_blur = st.sidebar.checkbox("Blur (manual conv)", value=False)
    blur_size = st.sidebar.selectbox("Blur kernel size", [3,5,7], index=0) if apply_blur else 3
    apply_sharp = st.sidebar.checkbox("Sharpen (manual conv)", value=False)
    # Background removal
    st.sidebar.header("Background removal (optional)")
    do_bkg = st.sidebar.checkbox("Enable background removal", value=False)
    bkg_method = st.sidebar.selectbox("Method", ["HSV threshold","GrabCut"]) if do_bkg else None
    # Now process: apply transform then filters then bkg removal (order: transform -> filter -> bkg)
    st.subheader("Preview & Controls")
    st.markdown("Left: original image. Right: current pipeline result.")
    # apply transform
    transformed_img = apply_homography_to_image(img, H, dst_size=dst_size, border_mode=cv2.BORDER_REPLICATE)
    # Filters
    filtered_img = transformed_img.copy().astype(np.float32)
    if apply_blur:
        k = blur_kernel(blur_size)
        filtered_img = manual_convolve(filtered_img, k)
    if apply_sharp:
        s = sharpen_kernel()
        filtered_img = manual_convolve(filtered_img, s)
    # Background removal applied to filtered image (foreground kept)
    bkg_mask = None
    bkg_removed = filtered_img.astype(np.uint8)
    if do_bkg:
        if bkg_method == "HSV threshold":
            st.sidebar.markdown("HSV thresholds (use slider to find background color range to remove).")
            h_min = st.sidebar.slider("H min", 0, 179, 0)
            h_max = st.sidebar.slider("H max", 0, 179, 179)
            s_min = st.sidebar.slider("S min", 0, 255, 0)
            s_max = st.sidebar.slider("S max", 0, 255, 255)
            v_min = st.sidebar.slider("V min", 0, 255, 0)
            v_max = st.sidebar.slider("V max", 0, 255, 255)
            lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
            mask, bkg_removed = remove_background_hsv(filtered_img.astype(np.uint8), lower, upper)
            bkg_mask = mask
        else:
            st.sidebar.markdown("Drawn rectangle will be used as initial GrabCut rect. Adjust iter count if needed.")
            iter_count = st.sidebar.slider("GrabCut iterations", 1, 10, 5)
            rect = (int(0.05*dst_size[0]), int(0.05*dst_size[1]),
                    int(0.9*dst_size[0]), int(0.9*dst_size[1]))
            mask, bkg_removed = grabcut_bkg_removal(filtered_img.astype(np.uint8), rect=rect, iter_count=iter_count)
            bkg_mask = mask
    # Show images side by side
    show_side_by_side(img, bkg_removed)
    st.subheader("Matrix & Download")
    st.markdown("Applied 3×3 homography (H):")
    st.write(H)

    result_pil = pil_from_cv(bkg_removed)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download result (PNG)", data=byte_im, file_name="result.png", mime="image/png")
    st.stop()

    # show mask if exists
    if bkg_mask is not None:
        st.subheader("Background mask preview")
        st.image(Image.fromarray(bkg_mask.astype(np.uint8)), use_column_width=False, width=400)

# Page: Team Members
elif page == "Team Members":
    st.title("Team Members")
num_members = st.number_input("How many team members?", min_value=1, max_value=10, value=1)
members = []
st.write("---")

for i in range(int(num_members)):
    st.subheader(f"Member {i+1}")

    name = st.text_input(f"Name for member {i+1}", key=f"name_{i}")
    role = st.text_input(f"Role for member {i+1}", key=f"role_{i}")
    photo = st.file_uploader(f"Upload photo for member {i+1}",
                              type=["jpg", "jpeg", "png"],
                              key=f"photo_{i}")

    members.append((name, role, photo))
    st.write("")

st.write("## Preview Team Profile")
st.write("---")

for i, (name, role, photo) in enumerate(members):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if photo:
            st.image(photo, width=150, caption=name or f"Member {i+1}")
        else:
            st.info("No photo uploaded")

    with col2:
        st.markdown(f"**Name:** {name or 'Not specified'}")
        st.markdown(f"**Role:** {role or 'Not specified'}")

    st.write("---")
st.stop()

st.markdown("**How the app works (short):**")
st.write("""
    1. Upload an image, choose transforms and filters.
    2. Each transform is built as a 3×3 homogeneous matrix (provided above).
    3. The matrix is applied to the image via a homography (warpPerspective).
    4. Blur and sharpen use explicit convolution loops (no OpenCV blur).
    5. Background removal optional via HSV threshold or GrabCut.
    """)
