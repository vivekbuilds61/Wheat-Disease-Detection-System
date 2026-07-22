import rasterio
import numpy as np

def calculate_ndvi(nir_path, red_path, output_path):
    with rasterio.open(nir_path) as nir:
        nir_band = nir.read(1).astype(float)

    with rasterio.open(red_path) as red:
        red_band = red.read(1).astype(float)

    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-6)

    profile = nir.profile
    profile.update(dtype=rasterio.float32, count=1)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)

    print("✅ NDVI saved:", output_path)

if __name__ == "__main__":
    calculate_ndvi(
        "data/satellite/NIR.tif",
        "data/satellite/RED.tif",
        "data/satellite/ndvi_output.tif"
    )
