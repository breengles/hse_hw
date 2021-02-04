#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

typedef pair<int, int> intpair;
int ans = INT_MAX;

struct DSet
{
    int *p, *r;

    DSet(int n)
    {
        this -> p = new int[n];
        this -> r = new int[n];

        for (int i = 0; i < n; i++)
        {
            r[i] = 0;
            p[i] = i;
        }
    }

    int get(int u)
    {
        if (u != p[u])
            p[u] = get(p[u]);
        return p[u];
    }

    void link(int c1, int c2)
    {
        if (r[c2] > r[c1])
            p[c2] = c1;
        else
            p[c1] = c2;

        if (r[c1] == r[c2])
            r[c2]++;
    }
};



void dfs(int v, vector<pair<int, int>> g[], vector<int> &d, int p = -1)
{
    for (auto u = g[v].begin(); u != g[v].end(); u++)
    {
        int idx = u -> first;
        if (p != idx)
            {
                d[idx] = max(d[v], u -> second);
                dfs(idx, g, d, v);
            }
    }
}


int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int h, n;
    cin >> h >> n;

    pair<int, int> s;
    vector<pair<int, int>> ver(3 * n);
    for (int i = 0; i < n; i++)
    {
        int x, y;
        cin >> x >> y;
        ver[i] = make_pair(x, y);
        ver[i + n] = make_pair(x, 0);
        ver[i + 2 * n] = make_pair(x, h);
    }
    
    vector<pair<int, intpair>> edges;
    for (int i = 0; i < 3 * n; i++)
        for (int j = 0; j < 3 * n; j++)
        {
            int w = pow(ver[i].first - ver[j].first, 2) + pow(ver[i].second - ver[j].second, 2);
            edges.push_back(make_pair(w, make_pair(i, j)));
        }

    DSet set(3 * n);
    sort(edges.begin(), edges.end());

    vector<pair<int, int>> mst[3 * n];

    for (auto i = edges.begin(); i != edges.end(); i++)
    {
        int u = i -> second.first;
        int v = i -> second.second;

        int su = set.get(u);
        int sv = set.get(v);

        if (su != sv)
        {
            mst[u].push_back(make_pair(v, i -> first));
            mst[v].push_back(make_pair(u, i -> first));
            set.link(su, sv);
        }
    }

    for (int i = n; i < 2 * n; i++)
    {
        vector<int> d(3 * n, 0);
        dfs(i, mst, d);
        for (int j = 0; j < n; j++)
            ans = min(ans, d[j + 2 * n]);
    }
    printf("%.9lf", sqrt(ans));

    return 0;
}