#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

bool dfs(int v, vector<int> g[], int p[], bool visited[], bool proc[], int &last){

    if (!visited[v]){
        visited[v] = true;
        proc[v] = true;

        for (vector<int>::iterator u = g[v].begin(); u != g[v].end(); u++){
            p[*u] = v;
            if (!visited[*u] && dfs(*u, g, p, visited, proc, last)){
                return true;
            }
            else if (proc[*u]){
                last = v;
                return true;
            }
        }
    }
    proc[v] = false;
    return false;
}



int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n, m;
    cin >> n >> m;

    vector<int> g[n];

    for (int i = 0; i < m; i++)
    {
        int a, b;
        cin >> a >> b;
        g[a - 1].push_back(b - 1);
    }

    bool visited[n];
    bool processing[n];
    int p[n];
    for (int i = 0; i < n; i++)
    {
        visited[i] = false;
        processing[i] = false;
        p[i] = -1;
    }


    vector<int> ans;
    for (int v = 0; v < n; v++)
    {
        if (!visited[v]){
            int last;
            if (dfs(v, g, p, visited, processing, last)){
                cout << "YES\n";

                int u = p[last];
                ans.push_back(last);
                while (u != last)
                {
                    ans.push_back(u);
                    u = p[u];
                }
                for (vector<int>::iterator i = ans.end() - 1; i >= ans.begin(); i--)
                    cout << *i + 1 << " ";    
                break;
            }
        }
        if (v == n - 1)
            cout << "NO";
    }

    return 0;
}