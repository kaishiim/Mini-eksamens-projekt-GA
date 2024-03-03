import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig = plt.subplots()
plt.plot(x, y)
st.pyplot(fig)