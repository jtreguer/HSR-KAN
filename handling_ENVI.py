import rasterio
import matplotlib.pyplot as plt
import numpy as np
import spectral
# Specify the path to the ENVI data file and the file with .hdr
file = '/mnt/c/data/chikusei_ENVI/HyperspecVNIR_Chikusei_20140729.bsq'
header_file = '/mnt/c/data/chikusei_ENVI/HyperspecVNIR_Chikusei_20140729.hdr'

# Open the ENVI image using rasterio
with rasterio.open(file) as src:
    # Read the hyperspectral data into a NumPy array
    print("Reading HS data")
    hyperspectral_data = src.read()

    # Display information about the hyperspectral data
    print('Shape of hyperspectral data:', hyperspectral_data.shape)
    print('Number of bands:', src.count)
    
#Here we can see the wavelengths of the data
img = spectral.open_image(header_file)

# Access the wavelengths associated with each band
wavelengths = img.bands.centers

# Display information about the hyperspectral data and wavelengths
print('Shape of hyperspectral data:', img.shape)
print('Number of bands:', img.shape[2])
print('Wavelengths:', wavelengths)
    # You can now work with the hyperspectral data using NumPy operations

#Let's show specific wavelengths
ind = wavelengths.index(0.46064)
plt.imshow(hyperspectral_data[ind,:,:])
plt.show()

ind = wavelengths.index(0.56386)
plt.imshow(hyperspectral_data[ind,:,:])
plt.show()

#Let's combine a short, middle and long wavelength
img = np.zeros([img.shape[0], img.shape[1], 3], np.float32)

ind1 = wavelengths.index(0.46064)
ind2 = wavelengths.index(0.56386)
ind3 = wavelengths.index(0.67224)

img[:,:,2] = hyperspectral_data[ind1,:,:]
img[:,:,1] = hyperspectral_data[ind2,:,:]
img[:,:,0] = hyperspectral_data[ind3,:,:]
plt.imshow(img)
plt.show()