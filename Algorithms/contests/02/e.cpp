#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

#define inf INT_MAX

void dfs_like(int v, int **g, int n, vector<bool> &visited, int cnt, int w, int &ans, vector<int> &prev, int &last){
    // if (cnt == n && g[v][0] != 0){
    if (cnt == n){
        if (w <= ans){
            ans = w;
            last = v;
            cout << "last = " << last << "\n";
        }
        return;
    }

    cout << "v = " << v << "\n";
    for (int u = 0; u < n; u++){
        if (!visited[u] && g[v][u] != 0){
            prev[u] = v;
            cout << "u = " << u << " <- " << prev[u] << "\n";
            visited[u] = true;
            dfs_like(u, g, n, visited, cnt + 1, w + g[v][u], ans, prev, last);
            visited[u] = false;
            // prev.clear();
        }
    }
}



int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n;
    cin >> n;
    int **g;
    g = new int *[n];
    for (int i = 0; i < n; i++){
        g[i] = new int[n];
    }

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++)
            cin >> g[j][i];
    }

    int nums = pow(2,n);
    int d[nums][n];


    for (int v = 0; v < n; v++){
        for (int s = 0; s < nums; s++)
            d[nums][v] = inf;
    }
    d[1][0] = 0;

    // for (int s = 2; s < nums; s++)
    //     d[s][0] = inf;


    for (int s = 1; s < nums; s++){
        if ((s >> 0) & 1){
            for (int v = 0; v < n; v++){
                if ((s>>v) & 1){
                    for (int u = 0; u < n; u++){
                        if ((s >> u) & 1)
                            d[s][v] = min(d[s][v], d[s - (1 << v)][u] + g[u][v]);
                    }
                }
            }
        }
    }

    int ans = inf;
    for (int u = 0; u < n; u++){
        ans = min(ans, d[nums - 1][u]);
    }
    cout << ans;


    // vector<bool> visited(n);
    // visited[0] = true;
    // for (int i = 1; i < n; i++)
    //     visited[i] = false;

    // int ans = inf;
    // int cnt = 1;
    // int w = 0;
    // int v = 0;

    // vector<int> prev(n);
    // int last = -1;
    // for (int i = 0; i < n; i++)
    //     prev[i] = -1;
    // dfs_like(v, g, n, visited, cnt, w, ans, prev, last);


    // for (int i = 0 ; i<n;i++){
    //     cout << "i = " << i << " | prev[i] = " << prev[i] << "\n";
    // }


    // cout << ans << "\n";
    // int curr = last;
    // // cout << last << " ";
    // while(curr != 0){
    //     cout << curr << " ";
    //     curr = prev[curr];
    // }

    return 0;
}