CUDA_VISIBLE_DEVICES=0 python entry_point.py sde=vp dataset=mnist model=concatsquashmixer2d hydra.run.dir=./checkpoint/ training=cnf