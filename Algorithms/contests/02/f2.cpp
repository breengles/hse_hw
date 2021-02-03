#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long


void dfs_like(int v, vector<int> g[], vector<int> &d, bool visited[]){
    visited[v] = true;
    for (vector<int>::iterator u = g[v].begin(); u != g[v].end(); u++){
        if (!visited[*u])
            dfs_like(*u, g, d, visited);
        d[v] = max(d[v], 1 + d[*u]);
        // cout << "v = " << v << " | d[v] = " << d[v] << "\n";
    }
}


int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);

    assert(freopen("longpath.in", "r", stdin));
    assert(freopen("longpath.out ", "w", stdout));

    int n, m;
    cin >> n >> m;

    vector<int> g[n];

    for (int i = 0; i < m; i++){
        int a, b;
        cin >> a >> b;
        g[a - 1].push_back(b - 1);
    }

    vector<int> d(n + 1);
    bool visited[n + 1];
    for (int i = 0; i <= n; i++){
        d[i] = 0;
        visited[i] = false;
    }


    for (int i = 0; i < n; i++){
        if (!visited[i])
            dfs_like(i, g, d, visited);
    }

    cout << *max_element(d.begin(), d.end());
    
    return 0;
}