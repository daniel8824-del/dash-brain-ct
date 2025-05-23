###### **This app's features**

This app displays 3D brain CT scan data from patients with traumatic brain injury (TBI). The images show various types of [intracranial hemorrhage (ICH)](https://en.wikipedia.org/wiki/Intracranial_hemorrhage) including subarachnoid hemorrhage, intraparenchymal hemorrhage, intraventricular hemorrhage, and intraaxial hemorrhage. The purpose of this app is to demonstrate the extraction of spatial range of hemorrhage areas and automated hemorrhage detection and segmentation techniques.

This app's images are displayed using [`dash-slicer`](https://dash.plotly.com/slicer). This is a Dash component that provides an interactive 3D slicing view of volume data. This app also provides an AI medical assistant chatbot feature to help with brain CT analysis and hemorrhage type learning.

###### **Data Source and Citation**

The CT scan data used in this application comes from the **PhysioNet CT-ICH Dataset**:

> **Hssayeni, M. (2020).** Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation (version 1.3.1). PhysioNet. https://doi.org/10.13026/4nae-zg36

This dataset contains 82 CT scans from patients with traumatic brain injury, collected from Al Hilla Teaching Hospital, Iraq, with proper ethical approval and anonymization.

###### **Reference**

1. **Hssayeni, M. D., Croock, M. S., Salman, A. D., Al-khafaji, H. F., Yahya, Z. A., & Ghoraani, B. (2020).** Intracranial Hemorrhage Segmentation Using A Deep Convolutional Model. *Data*, 5(1), 14.

2. **Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000).** PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation* [Online]. 101 (23), pp. e215–e220.

---

**⚠️ Disclaimer**: This tool is for educational purposes only and should not be used for actual clinical diagnosis. All medical decisions must be made in consultation with qualified medical professionals.