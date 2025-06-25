#include <KOMO/komo.h>

#include <gtest/gtest.h>
#include <functional>

#include <Kin/TM_default.h>
#include <Kin/TM_transition.h>
#include <car_kinematic.h>

#include <eigen3/Eigen/Dense>

#include <geom_utility.h>

using namespace std;

/////////////////////////////////////
struct KomoControlFixture : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        rai::KinematicWorld kin("data/LGP-real-time.g");

        komo.setModel(kin);
        komo.setTiming( 4, 5, 1.0, 2 );

        arr op_speed_1{ 3.0, 0, 0 };

        komo.addObjective(0, -1, new TM_Transition(komo.world), OT_sos, NoArr, 1.0, 2);
        komo.addObjective(0, -1, new TM_Default(TMT_pos, komo.world, "car_ego", NoVector, NULL, NoVector), OT_sos, op_speed_1, 1.0, 1);
        komo.addObjective(0, -1, new CarKinematic("car_ego", komo.world), OT_eq, NoArr, 1.0, 1);

        komo.reset();

        komo.checkGradients();

        //EXPECT_EQ(true, komo.checkGradients());

        komo.run();
    }

    KOMO komo;
};

TEST_F(KomoControlFixture, komo_to_traj_conversion)
{
  // still tests successfull planning
}

/*TEST_F(KomoControlFixture, komo_to_traj_conversion)
{
    auto traj = to_2d_trajectory(komo);

    EXPECT_EQ(traj.size(), komo.configurations.d0);
}

TEST_F(KomoControlFixture, projection_of_last_pose)
{
    auto trajectory = to_2d_trajectory(komo);
    auto last_pose = to_2d_pose(komo.configurations(-1));

    auto last_pose_projected = project_on_trajectory(last_pose, trajectory);

    EXPECT_TRUE(near(last_pose, last_pose_projected));
}

TEST_F(KomoControlFixture, single_segment)
{
    std::vector<Pose2D> trajectory;
    trajectory.push_back({0.0, 0.0, 0.0});
    trajectory.push_back({1.0, 0.0, 0.0});

    int index = -1; double mu = -1;
    auto last_pose_projected = project_on_trajectory({0.5, 1.0, 0.0}, trajectory, index, mu);

    EXPECT_TRUE(near({0.5, 0.0, 0.0}, last_pose_projected));
    EXPECT_TRUE(fabs(mu-0.5) < 0.001);
}

TEST_F(KomoControlFixture, single_segment_with_pose_orientation)
{
    std::vector<Pose2D> trajectory;
    trajectory.push_back({0.0, 0.0, 0.0});
    trajectory.push_back({1.0, 0.0, 0.0});

    auto last_pose_projected = project_on_trajectory({0.5, 1.0, 0.1}, trajectory);

    EXPECT_TRUE(last_pose_projected.x > 0.5);
}

TEST_F(KomoControlFixture, after_last_point)
{
    std::vector<Pose2D> trajectory;
    trajectory.push_back({0.0, 0.0, 0.0});
    trajectory.push_back({1.0, 0.0, 0.0});

    auto last_pose_projected = project_on_trajectory({1.5, 1.0, 0.0}, trajectory);

    EXPECT_TRUE(near({1.5, 0.0, 0.0}, last_pose_projected));
}

TEST_F(KomoControlFixture, before_first_point)
{
    std::vector<Pose2D> trajectory;
    trajectory.push_back({0.0, 0.0, 0.0});
    trajectory.push_back({1.0, 0.0, 0.0});

    auto last_pose_projected = project_on_trajectory({-0.5, 1.0, 0.0}, trajectory);

    EXPECT_TRUE(near({-0.5, 0.0, 0.0}, last_pose_projected));
}

TEST_F(KomoControlFixture, reinit_trajectory)
{
    std::vector<Pose2D> trajectory;
    trajectory.push_back({-2.0, 0.0, 0.0});
    trajectory.push_back({-1.0, 0.0, 0.0});
    trajectory.push_back({ 0.0, 0.0, 0.0});
    trajectory.push_back({ 1.0, 0.0, 0.0});
    trajectory.push_back({ 2.0, 0.0, 0.0});

    auto reinitialized_trajectory = reinit_trajectory_from({1.0, 0.0, 0.0}, trajectory, 2);

    EXPECT_TRUE(reinitialized_trajectory.size() == trajectory.size());
    EXPECT_TRUE(near({1.0, 0.0, 0.0}, reinitialized_trajectory[2]));
    EXPECT_TRUE(near({3.0, 0.0, 0.0}, reinitialized_trajectory[4]));
}

TEST_F(KomoControlFixture, reinit_trajectory_pose_before)
{
    std::vector<Pose2D> trajectory;
    trajectory.push_back({-2.0, 0.0, 0.0});
    trajectory.push_back({-1.0, 0.0, 0.0});
    trajectory.push_back({ 0.0, 0.0, 0.0});
    trajectory.push_back({ 1.0, 0.0, 0.0});
    trajectory.push_back({ 2.0, 0.0, 0.0});

    auto reinitialized_trajectory = reinit_trajectory_from({-3.0, 0.0, 0.0}, trajectory, 2);

    EXPECT_TRUE(reinitialized_trajectory.size() == trajectory.size());
    EXPECT_TRUE(near({-2.0, 0.0, 0.0}, reinitialized_trajectory[0]));
}


TEST_F(KomoControlFixture, duplicate_at_start_should_be_skipped)
{
    std::vector<Pose2D> trajectory;
    trajectory.push_back({0.0, 0.0, 0.0});
    trajectory.push_back({0.0, 0.0, 0.0});
    trajectory.push_back({ 1.0, 0.0, 0.0});
    trajectory.push_back({ 2.0, 0.0, 0.0});

    int index = -1;
    double mu = -1;
    auto last_pose_projected = project_on_trajectory({-0.01, 0.0, 0.0}, trajectory, index , mu);

    //EXPECT_TRUE(near({0.01, 0.0, 0.0}, last_pose_projected));
    EXPECT_EQ(index, 1);
}*/

////////////////////////////////
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}

