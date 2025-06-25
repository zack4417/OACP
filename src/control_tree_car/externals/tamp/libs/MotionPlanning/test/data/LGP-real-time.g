### objs

#vehicles
body car_ego { type=9 rel=<T t(1.25 0.0 0.05)> size=[4.3 1.7 1.0 .02]  color=[0 1 1] contact }

#road
body ground{ type=9, X=<T t(0.0 -0.00 0.0)>, size=[50. 3.0 .04 .01], color=[.3 .3 .3] }

## Joints
joint (ground car_ego)         { from=<T t( 0.0   0.00 .5 ) t(0 0 0)> to=<T > type=8 agent=true }
