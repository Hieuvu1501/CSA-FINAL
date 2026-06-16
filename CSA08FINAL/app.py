import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 1. CẤU HÌNH HỆ THỐNG

st.set_page_config(
    page_title="AI & RAM Economics",
    page_icon="🧠",
    layout="wide"
)


# 2. HÀM XỬ LÝ DỮ LIỆU (CORE)

def clean_currency(x):
    """Chuyển đổi giá tiền từ chuỗi (ví dụ '$1,000') sang số float"""
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        clean_str = x.replace('$', '').replace(',', '')
        try:
            return float(clean_str)
        except ValueError:
            return 0
    return 0

def clean_ram(x):
    """Lấy số từ chuỗi RAM (ví dụ '16GB' -> 16.0)"""
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        nums = ''.join(filter(str.isdigit, x))
        return float(nums) if nums else 0
    return 0

def identify_maker(row):
    """Xác định hãng sản xuất chip (NVIDIA hay AMD) từ tên và chipset"""
    full_text = (str(row.get('name', '')) + " " + str(row.get('chipset', ''))).upper()
    
    if 'NVIDIA' in full_text or 'GEFORCE' in full_text:
        return 'NVIDIA'
    elif 'AMD' in full_text or 'RADEON' in full_text:
        return 'AMD'
    return 'Other'


# 3. THANH ĐIỀU HƯỚNG

st.sidebar.title("Nội dung báo cáo")
page = st.sidebar.radio("Chọn phần thuyết trình:", 
    ["1. Tổng quan dự án", 
     "2. Lịch sử giá RAM (Vĩ mô)", 
     "3. Công cụ tính giá AI (Chung)",
    #  "4. So sánh NVIDIA vs AMD (Thống kê)",
     "4. Dự báo giá: Cuộc chiến Model"] 
)

st.sidebar.markdown("---")
st.sidebar.markdown("Dữ Liệu nguồn:")
st.sidebar.markdown("""
*[ram_price.csv]* *(Dữ liệu lịch sử giá RAM)*

*[gpu_specs_prices.csv]* *(Dữ liệu thị trường GPU)*
""")



# TRANG 1: TỔNG QUAN (OVERVIEW)

if page == "1. Tổng quan dự án":
    st.title("🧠 The Cost of Intelligence")
    st.subheader("Phân tích tác động kinh tế của AI lên thị trường phần cứng")
    

    col1, col2, col3 = st.columns(3)
    col1.metric("Nhu cầu RAM (LLM)", "cao gấp 5 lần", "So với Game thông thường")
    col2.metric("Xu hướng giá", "Tăng mạnh", "Do khan hiếm VRAM")
    col3.metric("Phương pháp", "Linear Regression", "Hồi quy tuyến tính")
    
    st.divider()
    

    try:
        st.image("https://gamingbolt.com/wp-content/uploads/2025/04/nvidia-rtx-5000.jpg", caption="Phần cứng AI là 'nhiên liệu' của kỷ nguyên mới", use_container_width=True)
    except FileNotFoundError:
        st.image("https://via.placeholder.com/800x400?text=AI+Hardware+Economics", use_container_width=True)



# TRANG 2: LỊCH SỬ (MOORE'S LAW)

elif page == "2. Lịch sử giá RAM (Vĩ mô)":
    st.title("📉 Lịch sử: Định luật Moore vs Thực tế")
    st.markdown("Kiểm chứng xu hướng giảm giá của bộ nhớ RAM qua 60 năm.")
    
    try:
        df = pd.read_csv('ram_price.csv')
        
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(df['date'], df['price'], label='Giá thực tế ($/MB)', color='blue')
        
        
        model = LinearRegression()
        X = df[['date']]
        y_log = np.log(df['price']) 
        model.fit(X, y_log)
        trend = np.exp(model.predict(X))
        
        ax.plot(df['date'], trend, 'r--', label="Xu hướng lý thuyết (Moore's Law)", linewidth=2)
        
        ax.set_title("Sự sụt giảm giá RAM (1950 - 2015)")
        ax.set_xlabel("Năm")
        ax.set_ylabel("Giá USD (Log Scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.success("Kết luận: Giá RAM giảm theo cấp số nhân trong quá khứ. Tuy nhiên, AI đang tạo ra một điểm gãy (Structural Break) làm giá VRAM tăng trở lại.")

    except Exception as e:
        st.error(f"Lỗi đọc file ram_price.csv: {e}")


# TRANG 3: CÔNG CỤ TÍNH GIÁ GPU (CHUNG)

elif page == "3. Công cụ tính giá AI (Chung)":
    st.title("💸 AI GPU Cost Estimator")
    st.markdown("Mô hình hồi quy đơn: Dự đoán giá dựa trên lượng VRAM mong muốn.")
    
    try:
        df = pd.read_csv('gpu_specs_prices.csv', encoding='latin-1')
        df['VRAM'] = df['memory'].apply(clean_ram)
        df['Price'] = df['price'].apply(clean_currency)
        df = df[(df['Price'] > 0) & (df['VRAM'] > 0)]
        
        
        X = df[['VRAM']]
        y = df['Price']
        model = LinearRegression()
        model.fit(X, y)
        price_per_gb = model.coef_[0]
        
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("""
            **Tham khảo nhu cầu VRAM:**
            - 8GB: Chatbot nhỏ
            - 16GB: Vẽ ảnh AI (Stable Diffusion)
            - 24GB+: Train mô hình / Chatbot lớn
            """)
        with col2:
            user_vram = st.slider("Bạn cần bao nhiêu GB VRAM?", 4, 80, 24, step=4)
            pred_price = model.predict([[user_vram]])[0]
            
            st.metric(f"Giá GPU ước tính ({user_vram}GB)", f"${pred_price:,.0f}", 
                     delta=f"Khoảng ${price_per_gb:.0f} cho mỗi GB tăng thêm")
            
        
        st.subheader("Biểu đồ phân tán (Toàn thị trường)")
        fig, ax = plt.subplots(figsize=(10, 5))
        sc = ax.scatter(df['VRAM'], df['Price'], c=df['Price'], cmap='viridis', alpha=0.6)
        
        
        x_range = np.linspace(df['VRAM'].min(), df['VRAM'].max(), 100).reshape(-1, 1)
        ax.plot(x_range, model.predict(x_range), 'r--', label='Mô hình dự báo')
        
        ax.set_xlabel("VRAM (GB)")
        ax.set_ylabel("Giá ($)")
        plt.colorbar(sc, label="Giá")
        ax.legend()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Lỗi: {e}")


# TRANG 4: SO SÁNH THỐNG KÊ (NVIDIA VS AMD)

# elif page == "4. So sánh NVIDIA vs AMD (Thống kê)":
    # st.title(" NVIDIA vs AMD: Ai 'hời' hơn?")
    # st.markdown("Phân tích chỉ số P/P (Price to Performance) trung bình.")
    
    # try:
    #     df = pd.read_csv('gpu_specs_prices.csv', encoding='latin-1')
    #     df['VRAM'] = df['memory'].apply(clean_ram)
    #     df['Price'] = df['price'].apply(clean_currency)
    #     df = df[(df['Price'] > 0) & (df['VRAM'] > 0)]
        
    #     # Phân loại hãng
    #     df['Brand_Clean'] = df.apply(identify_maker, axis=1)
    #     df = df[df['Brand_Clean'].isin(['NVIDIA', 'AMD'])]
        
    #     # Tính giá trên mỗi GB
    #     df['Price_Per_GB'] = df['Price'] / df['VRAM']
        
    #     # Tính trung bình
    #     avg_nvidia = df[df['Brand_Clean'] == 'NVIDIA']['Price_Per_GB'].mean()
    #     avg_amd = df[df['Brand_Clean'] == 'AMD']['Price_Per_GB'].mean()
        
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.metric("NVIDIA (Trung bình)", f"${avg_nvidia:,.2f} / 1GB", delta="Đắt hơn")
    #     with col2:
    #         st.metric("AMD (Trung bình)", f"${avg_amd:,.2f} / 1GB", delta=f"-${avg_nvidia - avg_amd:,.2f}", delta_color="normal")
            
    #     # Biểu đồ so sánh
    #     fig, ax = plt.subplots(figsize=(10, 6))
        
    #     # Vẽ 2 nhóm màu khác nhau
    #     for brand, color in [('NVIDIA', 'green'), ('AMD', 'red')]:
    #         subset = df[df['Brand_Clean'] == brand]
    #         ax.scatter(subset['VRAM'], subset['Price'], c=color, label=brand, alpha=0.6, s=100)
            
    #     ax.set_xlabel("VRAM (GB)")
    #     ax.set_ylabel("Giá ($)")
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)
    #     st.pyplot(fig)
        
    # except Exception as e:
    #     st.error(f"Lỗi: {e}")


# TRANG 5: DỰ BÁO GIÁ (CUỘC CHIẾN MODEL) - MỚI

elif page == "4. Dự báo giá: Cuộc chiến Model":
    st.title("🤖 AI Price Battle: Dự báo chi tiết")
    st.markdown("### Chúng tôi xây dựng 2 mô hình Hồi quy riêng biệt cho từng hãng.")
    
    try:
        
        df = pd.read_csv('gpu_specs_prices.csv', encoding='latin-1')
        df['VRAM'] = df['memory'].apply(clean_ram)
        df['Price'] = df['price'].apply(clean_currency)
        df = df[(df['Price'] > 0) & (df['VRAM'] > 0)]
        df['Brand_Clean'] = df.apply(identify_maker, axis=1)
        
        
        df_nvidia = df[df['Brand_Clean'] == 'NVIDIA']
        df_amd = df[df['Brand_Clean'] == 'AMD']
        
        if df_amd.empty or df_nvidia.empty:
            st.error("Thiếu dữ liệu của một trong hai hãng để so sánh!")
            st.stop()

        
        model_nvidia = LinearRegression()
        model_nvidia.fit(df_nvidia[['VRAM']], df_nvidia['Price'])
        
        model_amd = LinearRegression()
        model_amd.fit(df_amd[['VRAM']], df_amd['Price'])
        
        
        st.subheader("Nhập cấu hình mong muốn:")
        target_vram = st.slider("Chọn dung lượng VRAM (GB)", 4, 48, 16, step=4)
        
       
        pred_nv = model_nvidia.predict([[target_vram]])[0]
        pred_am = model_amd.predict([[target_vram]])[0]
        diff = pred_nv - pred_am
        
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🟢 NVIDIA Forecast")
            st.metric("Giá dự kiến", f"${pred_nv:,.0f}")
        with col2:
            st.markdown("#### 🔴 AMD Forecast")
            st.metric("Giá dự kiến", f"${pred_am:,.0f}", delta=f"Rẻ hơn ${diff:,.0f}", delta_color="normal")
            
        st.success(f"💡 **Phân tích:** Nếu bạn chọn AMD cho cấu hình {target_vram}GB, bạn có thể tiết kiệm khoảng **{diff/pred_nv*100:.1f}%** ngân sách so với NVIDIA.")

        
        fig, ax = plt.subplots(figsize=(10, 6))
        
       
        x_range = np.linspace(4, 48, 100).reshape(-1, 1)
        
        
        ax.plot(x_range, model_nvidia.predict(x_range), color='green', linewidth=3, label='Xu hướng giá NVIDIA')
        ax.scatter(df_nvidia['VRAM'], df_nvidia['Price'], color='green', alpha=0.1) # Điểm mờ
        
        
        ax.plot(x_range, model_amd.predict(x_range), color='red', linewidth=3, linestyle='--', label='Xu hướng giá AMD')
        ax.scatter(df_amd['VRAM'], df_amd['Price'], color='red', alpha=0.1) # Điểm mờ
        
        ax.set_title("So sánh độ dốc tăng giá (Price Slope)")
        ax.set_xlabel("VRAM (GB)")
        ax.set_ylabel("Giá dự kiến ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi: {e}")





# python -m streamlit run app.py