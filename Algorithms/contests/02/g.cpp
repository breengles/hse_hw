#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

void dfs(int v, vector<int> g[], int comp[], int cnum, bool visited[]){
    visited[v] = true;
    comp[v] = cnum;
    for (vector<int>::iterator u = g[v].begin(); u != g[v].end(); u++){
        if (!visited[*u])
            dfs(*u, g, comp, cnum, visited);
    }
}


int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);

    assert(freopen("connect.in", "r", stdin));
    assert(freopen("connect.out ", "w", stdout));

    int n, m;
    cin >> n >> m;

    vector<int> g[n];
    for (int i = 0; i < m; i++){
        int a, b;
        cin >> a >> b;
        g[a - 1].push_back(b - 1);
        g[b - 1].push_back(a - 1);
    }

    bool visited[n];
    for (int v = 0; v < n; v++){
        visited[v] = false;
    }


    int cnum = 0;
    int comp[n];
    for (int v = 0; v < n; v++){
        if (!visited[v]){
            cnum++;
            dfs(v, g, comp, cnum, visited);
        }
    }

    cout << cnum << "\n";
    for (int v = 0; v < n; v++)
        cout << comp[v] << " ";


    return 0;
}