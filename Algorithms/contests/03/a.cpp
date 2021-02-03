#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

bool dfs(int v, vector<int> g[], int p[], int color[], int &last){
    color[v] = 1;

    for (vector<int>::iterator u = g[v].begin(); u != g[v].end(); u++){
        p[*u] = v;

        if (color[*u] == 1){
            last = v;
            return true;
        }

        if (color[*u] == 0)
        {
            if (dfs(*u, g, p, color, last))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
    color[v] = 2;
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

    // for (int v = 0; v < n; v++)
    // {
    //     cout << v + 1 << " : ";
    //     for (vector<int>::iterator u = g[v].begin(); u != g[v].end(); u++)
    //     {
    //         cout << *u + 1 << " ";
    //     }
    //     cout << "\n";
    // }
    


    
    int color[n];
    int p[n];
    for (int i = 0; i < n; i++)
    {
        color[i] = 0;
        p[i] = -1;
    }


    vector<int> ans;
    for (int v = 0; v < n; v++)
    {
        if (color[v] == 0){
            int last;
            if (dfs(v, g, p, color, last)){
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