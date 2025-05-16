import streamlit as st

st.title("Code Predict Dog")
st.write("Data Structures และ Algorithms ที่ใช้พัฒนาโปรแกรมมีอะไรบ้างและใช้ในส่วนไหนของโปรแกรม เขียนอธิบายเหตุผลทำไมต้องใช้ อธิบายการทำงาน และแสดงวิธีการคำนวณของ Algorithms")

st.header("Data Structures")
with st.container():
    
    st.header("1. list", divider=True)
    st.subheader("ใช้ในการเก็บข้อมูลแบบลำดับ เช่น:")
    code = '''
        all_predictions = []
        boxes = []
        futures = [executor.submit(...)]
        labels = [f"{pred['class']} ({pred['confidence']:.2f})" for pred in result]
    '''
    st.code(code, language='python')
    st.divider()
    st.header("2. dict", divider=True)
    st.subheader("ใช้เก็บข้อมูล key-value เช่น:")
    code = '''
       {
            "bbox": [x, y, w, h],
            "label": pred['class'],
            "confidence": pred['confidence']
        }
    '''
    st.code(code, language='python')
    st.subheader("ใช้เก็บข้อมูล key-value เช่น:", divider=False)
    st.divider()
    st.header("3. pandas.DataFrame", divider=True)
    st.subheader("ใช้จัดการข้อมูลจาก dogs_cleaned.csv:")
    code = '''
        df = pd.read_csv(...)
        df["Suitable_For"] = df.apply(...)
        df_get = df.loc[df["Breed Name"] == name_breed]
    '''
    st.code(code, language='python')
    st.divider()


st.header("Sorting Algorithm")
with st.container():
    st.subheader("ใช้ max(..., key=...) เพื่อเลือก prediction ที่มี confidence สูงสุด")
    code = '''
        max(group, key=lambda x: x["confidence"])
    '''
    st.code(code, language='python')
    st.subheader("นี่คือการทำ partial sort บน list (ซึ่งภายใน Python ใช้ Timsort) → ทำหน้าที่เทียบค่า confidence ในแต่ละกลุ่มพันธุ์ เพื่อเอาค่าที่แม่นที่สุด")
    st.divider()
    
st.header("Searching Algorithm ที่ใช้")
with st.container():
    st.subheader("ใช้ .loc[], .str.lower().eq(...) กับ pandas เพื่อค้นหาพันธุ์ที่ตรงกับ prediction")
    code = '''
        mask = df["Breed Name"].str.lower().eq(name_breed.lower())
        df_breed = df.loc[mask].iloc[0]
    '''
    st.code(code, language='python')
    st.subheader("เป็นการทำ linear search บน DataFrame (Series comparison → boolean mask)")
    st.divider()


st.header("AI Algorithm ที่ใช้")
with st.container():
    st.subheader("ไม่ได้ train เอง แต่เรียกใช้ model ผ่าน Roboflow API:")
    code = '''
        CLIENT.infer(file_obj, model_id="stanford-dogs-0pff9/3")
    '''
    st.code(code, language='python')
    st.subheader("model ที่ใช้คือ pretrained object detection model จาก Roboflow")
    st.subheader("Roboflow ด้านหลังอาจใช้ YOLOv5/YOLOv8 หรือแบบอื่น (แล้วแต่โครงการ)")
    st.divider()
