#Include = 'data/man_model.ors'

### objs

#vehicles
body car_ego { type=9 rel=<T t(0 0.0 0.05)> size=[.2 .1 .1 .003]  color=[0 1 1] }
body car_1  { type=9 rel=<T t(0 0.0 0.05)> size=[.2 .1 .1 .003]  color=[1 0 0] type=1 }
body car_2  { type=9 rel=<T t(0 0.0 0.05)> size=[.2 .1 .1 .003]  color=[0.9 0 0] type=1 }
body car_3  { type=9 rel=<T t(0 0.0 0.05)> size=[.2 .1 .1 .003]  color=[0.8 0 0] type=1 }
body car_4  { type=9 rel=<T t(0 0.0 0.05)> size=[.2 .1 .1 .003]  color=[0.7 0 0] type=1 }
body car_5  { type=9 rel=<T t(0 0.0 0.05)> size=[.2 .1 .1 .003]  color=[0.6 0 0] type=1 }
body car_6  { type=9 rel=<T t(0 0.0 0.05)> size=[.2 .1 .1 .003]  color=[0.5 0 0] type=1 }
body car_7  { type=9 rel=<T t(0 0.0 0.05)> size=[.2 .1 .1 .003]  color=[0.4 0 0] type=1 }

#road
body lane_2{ type=9, X=<T t(0.0  0.10 0.0)>, size=[7. .2 .04 .01], color=[.3 .3 .3] }
body line  { type=9, X=<T t(-2.5 0.0 0.0)>,   size=[2 .02  .041 .01],color=[1 1 1] }
body lane_1{ type=9, X=<T t(-1.0 -0.10 0.0)>, size=[5. .2 .04 .01], color=[.3 .3 .3] }

## Joints
joint (lane_1 car_ego)         { from=<T t( -1.25   0.01 .05 ) t(0 0 0)> to=<T > type=6 agent=true}
joint (lane_2 car_1)           { from=<T t( -3.0      0 .05 ) t(0 0 0)> to=<T > type=3 }
joint (lane_2 car_2)           { from=<T t( -2.3      0 .05 ) t(0 0 0)> to=<T > type=3 random_bounds=[0.3, 0, 0]}
joint (lane_2 car_3)           { from=<T t( -1.6      0 .05 ) t(0 0 0)> to=<T > type=3 random_bounds=[0.3, 0, 0]}
joint (lane_2 car_4)           { from=<T t( -0.9      0 .05 ) t(0 0 0)> to=<T > type=3 }
joint (lane_2 car_5)           { from=<T t( -0.2      0 .05 ) t(0 0 0)> to=<T > type=3 }
joint (lane_2 car_6)           { from=<T t(  0.5      0 .05 ) t(0 0 0)> to=<T > type=3 }
joint (lane_2 car_7)           { from=<T t(  1.2      0 .05 ) t(0 0 0)> to=<T > type=3 }
#joint (lane_2 car_op_2)          { from=<T t( -1.0      0 .05 ) t(0 0 0)> to=<T > type=6 }
#joint (tableC block_a)          { from=<T t( 0.15  0 .02 ) t(0 0 0)> to=<T > type=10 } 
#joint (block_a block_c)         { from=<T t( 0.0   0 .1  ) t(0 0 0)> to=<T > type=10 } 
