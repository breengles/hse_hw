#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long


struct Graph {
    struct Edge {
        int a, b;
        int f, c;
    };

    int s, t;
    vector<vector<int>> g;
    vector<Edge> edges;
    int cc;
    vector<int> used;

    Graph(int n, int s, int t) : s(s), t(t), g(n), cc(0), used(n, -1) {
    }

    void add(int a, int b, int c) {
        g[a].push_back(edges.size());
        edges.push_back({a, b, 0, c});
        g[b].push_back(edges.size());
        edges.push_back({b, a, 0, 0});
    }

    bool dfs(int v) {
        used[v] = cc;
        if (v == t)
            return 1;
        for (int idx : g[v])
        {
            auto &e = edges[idx];
            if (e.f < e.c && used[e.b] != cc && dfs(e.b))
            {
                e.f++;
                edges[idx ^ 1].f--;
                return 1;
            }
        }
        return 0;   
    }

    void dfs2(int v, vector<int> &p) {
        if (v == t)
            p.push_back(v);

        for (int idx : g[v])
        {
            auto &e = edges[idx];
            if (e.f == 1)
            {
                e.f--;
                dfs2(e.b, p);
                p.push_back(e.a);
                return;
            }
        }
    }


    void ff() {
        while(dfs(s) && cc < 2)
            cc++;
        
        if (cc == 2)
        {
            vector<int> p1, p2;
            dfs2(s, p1);
            dfs2(s, p2);

            cout << "YES\n";
            for (auto v = p1.rbegin(); v != p1.rend(); v++)
                cout << *v + 1 << " ";
            cout << "\n";
            for (auto v = p2.rbegin(); v != p2.rend(); v++)
                cout << *v + 1 << " ";
        }
        else
            cout << "NO";
        
    }
};



int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n, m, s, t;
    cin >> n >> m >> s >> t;

    Graph g(n, s - 1, t - 1);

    for (int i = 0; i < m; i++)
    {
        int a, b;
        cin >> a >> b;
        g.add(a - 1, b - 1, 1);
    }
    
    g.ff();

    return 0;
}