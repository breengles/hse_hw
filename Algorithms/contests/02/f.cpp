#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

#define inf INT_MIN

void topsort(int i, bool visited[], stack<int> &tops, vector<int> g[]){
    visited[i] = true;
    for (std::vector<int>::iterator u = g[i].begin(); u != g[i].end(); u++){
        if (!visited[*u]){
            topsort(*u, visited, tops, g);
        }
    }
    tops.push(i);
}

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    // assert(freopen("longpath.in", "r", stdin));
    // assert(freopen("longpath.out ", "w", stdout));


    int n, m;
    cin >> n >> m;

    vector<int> g[n];

    for (int i = 0; i < m; i++){
        int a, b;
        cin >> a >> b;
        g[a - 1].push_back(b - 1);
    }

    stack<int> tops;
    bool* visited = new bool[n];

    for (int i = 0; i < n; i++)
        visited[i] = false;
    
    for (int i = 0; i < n; i++){
        if (!visited[i])
            topsort(i, visited, tops, g);
    }

    int dist[n];
    int maxd = 0;
    for (int s = 0; s < n; s++){
        for (int i = 0; i < n; i++){
            dist[i] = inf;
        }
        dist[s] = 0;

        stack<int> tmp = tops;
        while(!tmp.empty()){
            int i = tmp.top();
            tmp.pop();
            if (dist[i] != inf){
                for (std::vector<int>::iterator u = g[i].begin(); u != g[i].end(); u++){
                    if (dist[*u] < dist[i] + 1)
                        dist[*u] = dist[i] + 1;
                }
            }
        }
        cout << "\n";
        for (int i = 0; i < n; i++)
            (dist[i] == inf) ? cout << "INF " : cout << dist[i] << " ";
    }

    // cout << max;

    return 0;
}