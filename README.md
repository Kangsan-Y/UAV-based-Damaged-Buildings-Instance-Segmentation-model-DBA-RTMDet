# UAV-based-Damaged-Buildings-Instance-Segmentation-model-DBA-RTMDet
This is based on RTMDet to construct identification of damaged buildings framework in post-earthquake UAV. All our code will be published later.

# Train Test Inference

python tools/train_damaged.py 

python tools/test_damaged.py 

python demo/inference_damaged.py

Before executing the program, you need to modify the paths related to the model, data, etc. You can refer to sh/slurm_train.sh and sh/slurm_inference.sh in the sh folder for details.

# Dynamic Display of Results

Dynamic presentation of extraction results of damaged buildings in large-scale post-earthquake scenarios.

![freecompress-2](https://github.com/user-attachments/assets/9dd52dbc-3dd8-455e-9087-6f799e85cbfe)
     Detection - bounding box
     
     Segmentation - imstance mask


# Environment Require

# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
addict                    2.4.0                    pypi_0    pypi
aliyun-python-sdk-core    2.15.1                   pypi_0    pypi
aliyun-python-sdk-kms     2.16.4                   pypi_0    pypi
blas                      1.0                         mkl  
brotli-python             1.0.9            py38h6a678d5_8  
bzip2                     1.0.8                h5eee18b_6  
ca-certificates           2024.7.2             h06a4308_0  
certifi                   2024.7.4         py38h06a4308_0  
cffi                      1.17.0                   pypi_0    pypi
charset-normalizer        3.3.2              pyhd3eb1b0_0  
click                     8.1.7                    pypi_0    pypi
colorama                  0.4.6                    pypi_0    pypi
contourpy                 1.1.1                    pypi_0    pypi
crcmod                    1.7                      pypi_0    pypi
cryptography              43.0.0                   pypi_0    pypi
cuda-cudart               11.7.99                       0    nvidia
cuda-cupti                11.7.101                      0    nvidia
cuda-libraries            11.7.1                        0    nvidia
cuda-nvrtc                11.7.99                       0    nvidia
cuda-nvtx                 11.7.91                       0    nvidia
cuda-runtime              11.7.1                        0    nvidia
cuda-version              12.6                          3    nvidia
cycler                    0.12.1                   pypi_0    pypi
ffmpeg                    4.3                  hf484d3e_0    pytorch
fightingcv-attention      1.0.0                    pypi_0    pypi
filelock                  3.14.0                   pypi_0    pypi
fonttools                 4.53.1                   pypi_0    pypi
freetype                  2.12.1               h4a9f257_0  
gmp                       6.2.1                h295c915_3  
gmpy2                     2.1.2            py38heeb90bb_0  
gnutls                    3.6.15               he1e5248_0  
idna                      3.7              py38h06a4308_0  
importlib-metadata        8.2.0                    pypi_0    pypi
importlib-resources       6.4.0                    pypi_0    pypi
intel-openmp              2023.1.0         hdb19cb5_46306  
jinja2                    3.1.4            py38h06a4308_0  
jmespath                  0.10.0                   pypi_0    pypi
jpeg                      9e                   h5eee18b_3  
kiwisolver                1.4.5                    pypi_0    pypi
lame                      3.100                h7b6447c_0  
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      3.0                  h295c915_0  
libcublas                 11.10.3.66                    0    nvidia
libcufft                  10.7.2.124           h4fbf590_0    nvidia
libcufile                 1.11.0.15                     0    nvidia
libcurand                 10.3.7.37                     0    nvidia
libcusolver               11.4.0.1                      0    nvidia
libcusparse               11.7.4.91                     0    nvidia
libdeflate                1.17                 h5eee18b_1  
libffi                    3.4.4                h6a678d5_1  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libiconv                  1.16                 h5eee18b_3  
libidn2                   2.3.4                h5eee18b_0  
libnpp                    11.7.4.75                     0    nvidia
libnvjpeg                 11.8.0.2                      0    nvidia
libpng                    1.6.39               h5eee18b_0  
libstdcxx-ng              11.2.0               h1234567_1  
libtasn1                  4.19.0               h5eee18b_0  
libtiff                   4.5.1                h6a678d5_0  
libunistring              0.9.10               h27cfd23_0  
libwebp-base              1.3.2                h5eee18b_0  
lz4-c                     1.9.4                h6a678d5_1  
markdown                  3.6                      pypi_0    pypi
markdown-it-py            3.0.0                    pypi_0    pypi
markupsafe                2.1.3            py38h5eee18b_0  
matplotlib                3.7.5                    pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
mkl                       2023.1.0         h213fc3f_46344  
mkl-service               2.4.0            py38h5eee18b_1  
mkl_fft                   1.3.8            py38h5eee18b_0  
mkl_random                1.2.4            py38hdb19cb5_0  
mmcv                      2.1.0                    pypi_0    pypi
mmdet                     3.3.0                     dev_0    <develop>
mmengine                  0.10.4                   pypi_0    pypi
model-index               0.1.11                   pypi_0    pypi
mpc                       1.1.0                h10f8cd9_1  
mpfr                      4.0.2                hb69a4c5_1  
mpmath                    1.3.0            py38h06a4308_0  
ncurses                   6.4                  h6a678d5_0  
nettle                    3.7.3                hbbd107a_1  
networkx                  3.1              py38h06a4308_0  
numpy                     1.24.3           py38hf6e8229_1  
numpy-base                1.24.3           py38h060ed82_1  
opencv-python             4.10.0.84                pypi_0    pypi
opendatalab               0.0.10                   pypi_0    pypi
openh264                  2.1.1                h4ff587b_0  
openjpeg                  2.5.2                he7f1fd0_0  
openmim                   0.3.9                    pypi_0    pypi
openssl                   3.0.14               h5eee18b_0  
openxlab                  0.1.1                    pypi_0    pypi
ordered-set               4.1.0                    pypi_0    pypi
oss2                      2.17.0                   pypi_0    pypi
packaging                 24.1                     pypi_0    pypi
pandas                    2.0.3                    pypi_0    pypi
pillow                    10.4.0           py38h5eee18b_0  
pip                       24.2             py38h06a4308_0  
platformdirs              4.2.2                    pypi_0    pypi
psutil                    6.1.0                    pypi_0    pypi
pycocotools               2.0.7                    pypi_0    pypi
pycparser                 2.22                     pypi_0    pypi
pycryptodome              3.20.0                   pypi_0    pypi
pygments                  2.18.0                   pypi_0    pypi
pyparsing                 3.1.2                    pypi_0    pypi
pysocks                   1.7.1            py38h06a4308_0  
python                    3.8.19               h955ad1f_0  
python-dateutil           2.9.0.post0              pypi_0    pypi
pytorch                   2.0.1           py3.8_cuda11.7_cudnn8.5.0_0    pytorch
pytorch-cuda              11.7                 h778d358_5    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2023.4                   pypi_0    pypi
pyyaml                    6.0.2                    pypi_0    pypi
readline                  8.2                  h5eee18b_0  
requests                  2.28.2                   pypi_0    pypi
rich                      13.4.2                   pypi_0    pypi
scipy                     1.10.1                   pypi_0    pypi
setuptools                60.2.0                   pypi_0    pypi
shapely                   2.0.5                    pypi_0    pypi
six                       1.16.0                   pypi_0    pypi
sqlite                    3.45.3               h5eee18b_0  
sympy                     1.12             py38h06a4308_0  
tabulate                  0.9.0                    pypi_0    pypi
tbb                       2021.8.0             hdb19cb5_0  
termcolor                 2.4.0                    pypi_0    pypi
terminaltables            3.1.10                   pypi_0    pypi
tk                        8.6.14               h39e8969_0  
tomli                     2.0.1                    pypi_0    pypi
torchaudio                2.0.2                py38_cu117    pytorch
torchtriton               2.0.0                      py38    pytorch
torchvision               0.15.2               py38_cu117    pytorch
tqdm                      4.65.2                   pypi_0    pypi
typing_extensions         4.11.0           py38h06a4308_0  
tzdata                    2024.1                   pypi_0    pypi
urllib3                   1.26.19                  pypi_0    pypi
wheel                     0.43.0           py38h06a4308_0  
xz                        5.4.6                h5eee18b_1  
yapf                      0.40.2                   pypi_0    pypi
zipp                      3.20.0                   pypi_0    pypi
zlib                      1.2.13               h5eee18b_1  
zstd                      1.5.5                hc292b87_2 
