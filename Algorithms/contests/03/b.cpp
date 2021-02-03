#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long


void bfs(queue<int> q, vector<int> g[], pair<int, int> ans[]){
    int v;
    while (!q.empty())
    {
        v = q.front();
        q.pop();
        for (vector<int>::iterator u = g[v].begin(); u != g[v].end(); u++)
        {
            if (ans[*u].first >= (ans[v].first + 1) || ans[*u].first == -1)
            {
                q.push(*u);
                ans[*u] = make_pair(ans[v].first + 1, min(ans[*u].second, ans[v].second));
            }
        }

    }
}


int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n, k, m;
    cin >> n >> k;
    
    pair<int, int> ans[n];
    for (int i = 0; i < n; i++)
        ans[i] = make_pair(-1, n);  // (d, #)
    
    queue<int> q;
    for (int i = 0; i < k; i++)
    {
        int a;
        cin >> a;
        q.push(a - 1);
        ans[a - 1] = make_pair(0, a - 1);
    }
    
    cin >> m;
    vector<int> g[n];
    for (int i = 0; i < m; i++)
    {
        int a, b;
        cin >> a >> b;
        if (find(g[a - 1].begin(), g[a - 1].end(), b - 1) == g[a - 1].end())
        {
            g[a - 1].push_back(b - 1);
            g[b - 1].push_back(a - 1);
        }
    }

    bfs(q, g, ans);

    for (int i = 0; i < n; i++)
        cout << ans[i].first << " ";
    cout << "\n";
    for (int i = 0; i < n; i++)
        cout << ans[i].second + 1 << " ";
    
    return 0;
}