#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

ui **g, **rg;
vector<int> p1, p2;

void dfs(int s, int t, bool visited[], int n, bool second)
{
    visited[s] = true;

    if (second)
    {
        p2.push_back(s + 1);
    }
    else
    {
        p1.push_back(s + 1);
    }

    for (int i = 0; i < n; i++)
        if (rg[s][i] < g[s][i] && !visited[i] && !visited[t])
        {
            rg[s][i]++;
            dfs(i, t, visited, n, second);
        }
}


bool bfs(int s, int t, int parent[], int n)
{
    bool visited[n];
    for (int i = 0; i < n; i++)
        visited[i] = false;

    queue <int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;
    
    while (!q.empty())
    { 
        int u = q.front();
        q.pop();
        for (int v = 0; v < n; v++)
            if (!visited[v] && rg[u][v] > 0)
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
    bool second = false;
    int max_flow = 0;
  
    while (bfs(s, t, parent, n) && max_flow < 2)
    {
        int flow = 1;

        for (int v = t; v != s; v = parent[v])
        { 
            int u = parent[v];
            rg[u][v] -= flow;
            rg[v][u] += flow;
        }
        max_flow += flow;
    }



    // cout << "----------\n";
    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         cout << rg[i][j] << " ";
    //     }
    //     cout << "\n";
    // }
    // cout << "----------\n";
    
    // cout << "----------\n";
    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         cout << g[i][j] << " ";
    //     }
    //     cout << "\n";
    // }
    // cout << "----------\n";
    


    if (max_flow == 2)
    {
        cout << "YES\n";
        bool visited[n];
        for (int i = 0; i < n; i++)
            visited[i] = false;
        dfs(s, t, visited, n, false);

        for (int i = 0; i < n; i++)
            visited[i] = false;
        dfs(s, t, visited, n, true);

        for (auto u = p1.begin(); u != p1.end(); u++)
            cout << *u << " ";
        cout << "\n";
        for (auto u = p2.begin(); u != p2.end(); u++)
            cout << *u << " ";







        // vector<int> ans1, ans2;
        // for (int u = t; u != -1; u = p1[u])
        //     ans1.push_back(u + 1);

        // for (int u = t; u != -1; u = p2[u])
        //     ans2.push_back(u + 1);

        // for (auto u = ans1.rbegin(); u != ans1.rend(); u++)
        //     cout << *u << " ";
        // cout << "\n";
        // for (auto u = ans2.rbegin(); u != ans2.rend(); u++)
        //     cout << *u << " ";
        
    }
    else
    {
        cout << "NO";
    }
    
}


int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n, m, s, t;
    cin >> n >> m >> s >> t;

    g = new ui *[n];
    for (int i = 0; i < n; i++)
        g[i] = new ui [n];

    rg = new ui *[n];
    for (int i = 0; i < n; i++)
        rg[i] = new ui [n];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            g[i][j] = 0;

    for (int i = 0; i < m; i++)
    {
        int v, u;
        cin >> v >> u;
        if (v != u)
            g[v - 1][u - 1]++;
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            rg[i][j] = g[i][j];
    
    ff(s - 1, t - 1, n);

    return 0;
}