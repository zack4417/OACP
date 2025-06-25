#include <belief_state.h>

double transitionProbability(const std::vector< double >& bsFrom, const std::vector< double >& bsTo)
{
  double p = 0;
  int n = 0;
  for(auto w=0; w<bsFrom.size();++w)
  {
    if(bsTo[w] > 0)
    {
      p += bsFrom[w] / bsTo[w];
      ++n;
    }
  }
  return p / n;
}
