module load cuda/11.8

conda create -n 598 python=3.9

conda activate 598

pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install setuptools==57.5.0
pip uninstall numpy
pip install numpy==1.23.5
pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118

pip install typed-argument-parser
pip install networkx==2.5
pip install open3d
pip install git+https://github.com/openai/CLIP.git
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html

pip install diffusers["torch"]

pip install packaging
pip install ninja
pip install flash-attn==2.5.9.post1 --no-build-isolation



cd /project/yehhh/UIUC/programs/PyRep/
wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz;
echo "export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" >> $HOME/.bashrc; 
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT" >> $HOME/.bashrc;
echo "export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT" >> $HOME/.bashrc;
source $HOME/.bashrc;
conda activate 598
pip install -r requirements.txt; pip install -e .; cd ..

git clone https://github.com/MohitShridhar/RLBench.git
cd RLBench; git checkout -b peract --track origin/peract; pip install -r requirements.txt; pip install -e .; cd ..;


cd 3d_diffuser_actor
pip install -e .
cd ..


cd UniDepth
pip install -e .
cd ..

# pip install tap


module load cuda/11.8
xvfb-run --auto-servernum bash online_evaluation_rlbench/test.sh