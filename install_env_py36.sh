#打印
echo "start install env"
pip uninstall tensorflow-gpu
pip install pyglet==1.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python==4.5.4.58 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install gym==0.10.4 -i https://pypi.tuna.tsinghua.edu.cn/simple  

python3 l_pole_36.py 