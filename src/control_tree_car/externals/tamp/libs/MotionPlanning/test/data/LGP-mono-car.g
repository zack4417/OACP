#Include = 'data/man_model.ors'

### objs

#vehicles
body car_ego { type=9 rel=<T t(0 0.0 0.05)> size=[.2 .1 .1 .02]  color=[0 1 1] }

#road
body lane_1{ type=9, X=<T t(0.0 -0.10 0.0)>, size=[5. .2 .04 .01], color=[.3 .3 .3] }
body line  { type=9, X=<T t(0.0 0.0 0.0)>,   size=[5. .02  .041 .01],color=[1   1  1] }
body lane_2{ type=9, X=<T t(0.0  0.10 0.0)>, size=[5. .2 .04 .01], color=[.3 .3 .3] }

## Joints
joint (lane_1 car_ego)         { from=<T t( -2.0   0.01 .05 ) t(0 0 0)> to=<T > type=6 agent=true }
#joint (lane_2 car_op)          { from=<T t( 1.8      0  .05 ) t(0 0 0)> to=<T >  type=3 }

