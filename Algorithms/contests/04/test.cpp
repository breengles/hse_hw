Тихон Воробьев, [03.12.20 23:42]
int find_flow(int index_node, int end, int flow, vector< vector<int> > &adjacency_list, vector< vector<int> > &edges, vector<bool> &visited)
{
  if(index_node == end)
    return flow;
  visited[index_node] = true;
  for(auto index_edge = adjacency_list[index_node].begin(); index_edge < adjacency_list[index_node].end(); ++index_edge)
  {
    vector<int> edge = edges[*index_edge];
    if(!visited[edge[0]])
    {
      if(edge[1] > edge[2])
      {
        int res = find_flow(edge[0], end, min(flow, edge[1] - edge[2]), adjacency_list, edges, visited);
        if(res > 0)
        {
          edges[*index_edge][2] += res;
          if ((*index_edge) % 2 == 0)
            edges[(*index_edge)+1][2] -= res;
          else
            edges[(*index_edge)-1][2] -= res;
          return res;
        }
      }
    }
  }
  return 0;
}
 
int find_path(int index_node, int end, vector< vector<int> > &adjacency_list, vector< vector<int> > &edges, vector<int> &path)
{
  path.push_back(index_node);
  if(index_node == end)
    return 0;
  for(auto index_edge = adjacency_list[index_node].begin(); index_edge < adjacency_list[index_node].end(); ++index_edge)
  {
    vector<int> edge = edges[*index_edge];
    if(edge[2] == 1)
    {
      edges[*index_edge][2] = 0;
      return find_path(edge[0], end, adjacency_list, edges, path);
    }
  }
  return 0;
}