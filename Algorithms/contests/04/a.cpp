#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

typedef pair<int, int> intpair;

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



int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n, m;
    cin >> n >> m;

    vector<pair<int, intpair>> edges(m);

    for (int i = 0; i < m; i++)
    {
        int b, e, w;
        cin >> b >> e >> w;
        edges.push_back(make_pair(w, make_pair(b - 1, e - 1)));
    }

    
    ll w = 0;
    DSet set(n);
    sort(edges.begin(), edges.end());

    for (vector<pair<int, intpair>>::iterator i = edges.begin(); i != edges.end(); i++)
    {
        int u = i -> second.first;
        int v = i -> second.second;

        int su = set.get(u);
        int sv = set.get(v);

        if (su != sv)
        {
            set.link(su, sv);
            w += i -> first;
        }
    }
    cout << w;
    return 0;
}