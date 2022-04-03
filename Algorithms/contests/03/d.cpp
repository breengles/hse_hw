#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

# define INF INT_MAX

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n, s, f;
    cin >> n >> s >> f;
    s--;
    f--;

    int **g;
    g = new int *[n];
    for (int i = 0; i < n; i++)
        g[i] = new int[n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> g[i][j];

    int d[n];
    bool set[n];

    for (int i = 0; i < n; i++)
    {
        d[i] = INF;
        set[i] = false;
    }
    d[s] = 0;

    for (int i = 0; i < n; i++)
    {
        int d_min = INF;
        int v;
        for (int k = 0; k < n; k++)
        {
            if (!set[k] && d[k] <= d_min)
            {
                d_min = d[k];
                v = k;
            }
        }
        set[v] = true;

        for (int u = 0; u < n; u++)
        {            
            if (!set[u] && u != v && g[v][u] > -1 && d[v] < INF && d[u] > d[v] + g[v][u])
                d[u] = d[v] + g[v][u];
        }
        
    }

    if (d[f] != INF)
        cout << d[f];
    else
        cout << -1;

    return 0;
}