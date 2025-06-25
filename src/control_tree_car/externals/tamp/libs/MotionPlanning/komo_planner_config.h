#pragma once

#include <future>
#include <Core/array.h>

struct KOMOPlannerConfig
{
  const rai::String beliefStateTag_  = "BELIEF_START_STATE";

  double kinEqualityWeight_  = 1e4;
  double secPerPhase_        = 10.;
  double maxConstraint_      = 10 * 0.8;
  double minMarkovianCost_   = 0;

  uint microSteps_     = 20; // per phase

  std::launch executionPolicy_ = std::launch::async;
};
