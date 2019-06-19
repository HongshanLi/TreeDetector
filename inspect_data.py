import streamlit as st
import matplotlib.pyplot as plt
import os
import numpy as np


d = '/home/hongshan/data/Trees/037185/037185-0'
rgba = "037185-0_RGB-Ir.tif"
dsm = '037185-0_DSM.tif'

epth = os.path.join(d, dsm)

rgbpth = os.path.join(d, rgba)

b = plt.imread(rgbpth)
c = b[:,:,0:2]
a = plt.imread(epth)
a_ = a.reshape(1250, 1250, 1)

c_ = np.concatenate([c, a_], axis=2)


st.image(c_)
st.image(a_)







