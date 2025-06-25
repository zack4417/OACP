#include <KOMO/komo.h>

#include <functional>

#include <Kin/TM_default.h>
#include <Kin/TM_transition.h>

#include <gtest/gtest.h>

using namespace std;

/////////////////////////////////////
/*
TEST(TestKomoReInit, single_scenario)
{
    rai::KinematicWorld kin("data/LGP-real-time.g");

    KOMO komo; komo.setModel(kin);
    komo.setTiming( 4, 5, 1.0, 2 );

    arr op_speed_1{ 3.0, 0, 0 };

    komo.addObjective(0, -1, new TM_Transition(komo.world), OT_sos, NoArr, 1.0, 2);
    komo.addObjective(0, -1, new TM_Default(TMT_pos, komo.world, "car_ego", NoVector, NULL, NoVector), OT_sos, op_speed_1, 1.0, 1);

    komo.reset();

    //komo.checkGradients();

    //EXPECT_EQ(true, komo.checkGradients());
    komo.run();

    komo.getReport(true);
    //komo.displayTrajectory(1.0, true, true);

    EXPECT_TRUE(true);
}

TEST(TestKomoReInit, change_scale_and_reoptimize)
{
    // step 1
    rai::KinematicWorld kin("data/LGP-real-time.g");

    KOMO komo; komo.setModel(kin);
    komo.setTiming( 4, 50, 1.0, 2 );

    arr op_speed_1{ 3.0, 0, 0 };

    auto acc = komo.addObjective(0, -1, new TM_Transition(komo.world), OT_sos, NoArr, 1.0, 2);
    auto speed = komo.addObjective(0, -1, new TM_Default(TMT_pos, komo.world, "car_ego", NoVector, NULL, NoVector), OT_sos, op_speed_1, 1.0, 1);

    komo.reset();

    {
        komo.run();
    }

    //komo.getReport(true);

    // get previous optimization
    for(auto i = 0; i < acc->map->scale.d0; ++i)
    {
        acc->map->scale(i) *= 2.0;
    }

    // step 2
    {
        komo.run();
    }

    komo.getReport(true);


    EXPECT_TRUE(true);
}

TEST(TestKomoReInit, update_prefix_before_optim)
{
    // step 1
    rai::KinematicWorld kin("data/LGP-real-time.g");

    KOMO komo; komo.setModel(kin);

    const int steps = 50;
    komo.setTiming( 4, steps, 1.0, 2 );

    arr op_speed{ 3.0, 0, 0 };

    auto acc = komo.addObjective(0, -1, new TM_Transition(komo.world), OT_sos, NoArr, 1.0, 2);
    auto speed = komo.addObjective(0, -1, new TM_Default(TMT_pos, komo.world, "car_ego", NoVector, NULL, NoVector), OT_sos, op_speed, 1.0, 1);

    komo.reset(0);

    // update prefix
    komo.configurations(0)->q(0) = komo.configurations(2)->q(0) - 2 * op_speed(0) / steps;
    komo.configurations(0)->calc_Q_from_q();
    komo.configurations(0)->calc_fwdPropagateFrames();

    komo.configurations(1)->q(0) = komo.configurations(2)->q(0) - 1 * op_speed(0) / steps;
    komo.configurations(1)->calc_Q_from_q();
    komo.configurations(1)->calc_fwdPropagateFrames();
    //

    {
        komo.run();
    }

    komo.getReport(true);
    //komo.plotTrajectory();

    EXPECT_TRUE(true);
}*/


TEST(TestKomoReInit, update_prefix_before_optim_shift_traj_reopt)
{
    // step 1
    rai::KinematicWorld kin("data/LGP-real-time.g");

    KOMO komo; komo.setModel(kin);

    const int steps = 50;
    komo.setTiming( 4, steps, 1.0, 2 );

    arr op_speed{ 3.0, 0, 0 };

    auto acc = komo.addObjective(0, -1, new TM_Transition(komo.world), OT_sos, NoArr, 1.0, 2);
    auto speed = komo.addObjective(0, -1, new TM_Default(TMT_pos, komo.world, "car_ego", NoVector, NULL, NoVector), OT_sos, op_speed, 1.0, 1);

    komo.reset(0);

    // update prefix
    komo.configurations(0)->q(0) = komo.configurations(2)->q(0) - 2 * op_speed(0) / steps;
    komo.configurations(0)->calc_Q_from_q();
    komo.configurations(0)->calc_fwdPropagateFrames();

    komo.configurations(1)->q(0) = komo.configurations(2)->q(0) - 1 * op_speed(0) / steps;
    komo.configurations(1)->calc_Q_from_q();
    komo.configurations(1)->calc_fwdPropagateFrames();
    //

    {
        komo.run();
    }

    //
    komo.world.q(0) += 100;
    for(auto i = 0; i < komo.configurations.d0; ++i)
    {
        komo.configurations(i)->q(0) += 100;
        komo.configurations(i)->calc_Q_from_q();
        komo.configurations(i)->calc_fwdPropagateFrames();
    }
    // step 2
    {
        komo.reset(0);
        komo.run();
    }


    //komo.getReport(true);
    komo.plotTrajectory();

    EXPECT_TRUE(true);
}

////////////////////////////////
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
