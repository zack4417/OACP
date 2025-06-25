#include <geom_utility.h>

using namespace std;

uint getFrameIndex(const rai::KinematicWorld& G, const std::string& object_name)
{
  const auto object = G.getFrameByName( object_name.c_str() );
  for(auto i = 0; i < G.frames.size(); ++i )
  {
    if(object == G.frames(i))
    {
      return i;
    }
  }

  throw("object not found!");
}

/*
bool near(const Pose2D & a, const Pose2D & b, double eps)
{
    bool close = true;

    close = close && fabs(a.x-b.x) < eps;
    close = close && fabs(a.y-b.y) < eps;

    return close;
}

Pose2D project_on_trajectory(const Pose2D & p, std::vector<Pose2D> trajectory, int & index, double & mu)
{
    using namespace Eigen;

    index = -1;
    mu = -1;

    int I = 0; // number of passed valid points
    for(auto i = 0; i < trajectory.size()-1; ++i)
    {
        const auto a = trajectory[i];
        const auto b = trajectory[i+1];

        if(near(a, b))  // identical points, might be the prefix
            continue;

        Vector2D n {sin(p.yaw), -cos(p.yaw)};
        Vector2D ab {b.x - a.x, b.y - a.y};

        Matrix2f A;
        A << n.x, a.x - b.x,
             n.y, a.y - b.y;

        Vector2f B;
        B << a.x - p.x, a.y - p.y;

        if(fabs(A.determinant()) > 0.00001)
        {
            Vector2f S = A.inverse() * (B);
            double lambda = S[0];
            double _mu = S[1];

            if(0.0 <= _mu && _mu < 1.0
                    || I == 0 && _mu <= 1.0 // hack to make sure it works if the furst points were skipped
                    || i == trajectory.size() - 2 && 0 <= _mu) // proj found
            {
                index = i;
                mu = _mu;
                return {a.x + mu * ab.x, a.y + mu * ab.y, b.yaw};
            }
        }

        ++I;
    }

    return Pose2D{std::nan(""), std::nan(""), std::nan("")};
}


Pose2D project_on_trajectory(const Pose2D & p, std::vector<Pose2D> trajectory)
{
    int index; double mu;
    return project_on_trajectory(p, trajectory, index, mu);
}

std::vector<Pose2D> reinit_trajectory_from(const Pose2D & p, std::vector<Pose2D> trajectory, int prefix_size)
{
    std::vector<Pose2D> reinit_trajectory(trajectory.size());

    int index = 0;
    double mu = 0;
    auto proj = project_on_trajectory(p, trajectory, index, mu);

    if(index < prefix_size)
        return trajectory;

    int to_shift_ahead = index - prefix_size;

    for(auto i = 0; i < trajectory.size() - to_shift_ahead; ++i)
    {
        reinit_trajectory[i] = trajectory[i + to_shift_ahead];
    }

    for(auto i = trajectory.size() - to_shift_ahead; i < trajectory.size(); ++i)
    {
        reinit_trajectory[i].x = 2 * reinit_trajectory[i-1].x - reinit_trajectory[i-2].x;
        reinit_trajectory[i].y = 2 * reinit_trajectory[i-1].y - reinit_trajectory[i-2].y;
        reinit_trajectory[i].yaw = 2 * reinit_trajectory[i-1].yaw - reinit_trajectory[i-2].yaw;
    }

    return reinit_trajectory;
}

Pose2D to_2d_pose(rai::KinematicWorld * configuration)
{
    return {configuration->q(0), configuration->q(1), configuration->q(2)};
}

std::vector< Pose2D > to_2d_trajectory(const KOMO & komo)
{
    std::vector< Pose2D > trajectory(komo.configurations.d0);

    for(auto i = 0; i < komo.configurations.d0; ++i)
    {
        trajectory[i] = to_2d_pose(komo.configurations(i));
    }

    return trajectory;
}

Pose2D to_1d_pose(rai::KinematicWorld * configuration)
{
    return {configuration->q(0), 0, 0};
}

std::vector< Pose2D > to_1d_trajectory(const KOMO & komo)
{
    std::vector< Pose2D > trajectory(komo.configurations.d0);

    for(auto i = 0; i < komo.configurations.d0; ++i)
    {
        trajectory[i] = to_1d_pose(komo.configurations(i));
    }

    return trajectory;
}*/
