#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

# define INF 100*1000

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n;
    cin >> n;
    
    int **g;
    g = new int *[n];
    for (int i = 0; i < n; i++)
        g[i] = new int[n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            int a;
            cin >> a;
            if (a < 0)
                a = INF;
            g[i][j] = a;
        }

    int d[n][n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            d[i][j] = g[i][j];

    for (int k = 0; k < n; k++)
        for (int v = 0; v < n; v++)
            for (int u = 0; u < n; u++)
                d[v][u] = min(d[v][u], d[v][k] + d[k][u]);

    int diam = 0, rad = INF;
    for (int i = 0; i < n; i++)
    {
        int r = 0;
        for (int j = 0; j < n; j++)
        {
            if (d[i][j] != INF){
                diam = max(diam, d[i][j]);
                r = max(r, d[i][j]);
            }
        }
        rad = min(rad, r);
    }

    cout << diam << "\n" << rad;
    
    return 0;
}