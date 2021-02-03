#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

pair<int, int> **g, **rg;


void dfs(int s, bool visited[], int n)
{
    visited[s] = true;
    for (int i = 0; i < n; i++)
       if (rg[s][i].first > 0 && !visited[i])
           dfs(i, visited, n);
}


bool bfs(int s, int t, int parent[], int n)
{
    bool visited[n];
    for (int i = 0; i < n; i++)
    {
        visited[i] = false;
    }

    queue <int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;
  
    while (!q.empty())
    { 
        int u = q.front();
        q.pop();   
        for (int v = 0; v < n; v++)
            if (!visited[v] && rg[u][v].first > 0)
            { 
                q.push(v); 
                parent[v] = u; 
                visited[v] = true;
            } 
    } 
    return visited[t];
}


void ff(int s, int t, int n)
{ 
    int parent[n];
    int max_flow = 0;
  
    while (bfs(s, t, parent, n))
    { 
        int flow = INT_MAX;
        for (int v = t; v != s; v = parent[v])
        { 
            int u = parent[v]; 
            flow = min(flow, rg[u][v].first);
        } 

        for (int v = t; v != s; v = parent[v])
        { 
            int u = parent[v]; 
            rg[u][v].first -= flow;
            rg[v][u].first += flow;
        } 
        max_flow += flow;
    } 
    // cout << max_flow;

    bool visited[n];
    for (int i = 0; i < n; i++)
        visited[i] = false;
    dfs(s, visited, n);

    vector<int> ans;
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < n; j++) 
            if (visited[i] && !visited[j] && g[i][j].first > 0)
                ans.push_back(g[i][j].second);

    cout << ans.size() << " " << max_flow << "\n";
    sort(ans.begin(), ans.end());
    for (auto i = ans.begin(); i != ans.end(); i++)
        cout << *i + 1 << " ";
} 


int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n, m;
    cin >> n >> m;

    g = new pair<int, int> *[n];
    for (int i = 0; i < n; i++)
        g[i] = new pair<int, int> [n];

    rg = new pair<int, int> *[n];
    for (int i = 0; i < n; i++)
        rg[i] = new pair<int, int> [n];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            g[i][j].first = 0;

    for (int i = 0; i < m; i++)
    {
        int v, u, c;
        cin >> v >> u >> c;
        g[v - 1][u - 1] = {c, i};
        g[u - 1][v - 1] = {c, i};
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            rg[i][j] = g[i][j];


    ff(0, n-1, n);
    return 0;
}