#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

bool chk_bound(int x){
    return x >= 0 && x < 8;
}

int get_idx(int x, int y){
    return 8 * y + x;
}

pair<int, int> get_pair(int idx){
    return make_pair(idx % 8, idx / 8);
}

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);

    int first_step[2] = {2, -2};
    int second_step[2] = {1, -1};
    int n = 8;
    

    bool d[n][n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            d[i][j] = false;
    }
    
    string inp1, inp2;
    cin >> inp1 >> inp2;

    pair<int, int> start, stop;
    start = make_pair(char(inp1[0]) % 97, char(inp1[1]) % 49);
    stop = make_pair(char(inp2[0]) % 97, char(inp2[1]) % 49);
    d[start.first][start.second] = true;

    queue<pair<int, int>> q;
    q.push(start);

    int p[64];
    pair<int, int> cell;
    while(!q.empty())
    {
        cell = q.front();
        q.pop();
        if (cell == stop)
        {
            vector<int> ans;
            int v = get_idx(cell.first, cell.second);

            // cout << "ANSWER:\n";
            // cout << char(get_pair(v).first + 97) << get_pair(v).second + 1 << "\n";
            
            ans.push_back(v);
            while (v != get_idx(start.first, start.second))
            {
                v = p[v];

                // cout << char(get_pair(v).first + 97) << get_pair(v).second + 1 << "\n";
                
                ans.push_back(v);
            }
            reverse(ans.begin(), ans.end());
            for (vector<int>::iterator t = ans.begin(); t != ans.end(); t++)
            {
                pair<int, int> c = get_pair(*t);
                char x = c.first + 97;
                cout << x << c.second + 1 << "\n";
            }
            break;
        }
        for (int i = 0; i < 2; i++)
        {
            int x1 = cell.first + first_step[i];
            if (chk_bound(x1))
            {
                for (int j = 0; j < 2; j++)
                {
                    int y2 = cell.second + second_step[j];
                    // cout << "before: " << x1 + 1 << " " << y2 + 1 << "\n";
                    if (chk_bound(y2))
                    {
                        // cout << "final check: " << x1 + 1 << " " << y2 + 1 << "\n";
                        // cout << "visited? " << d[x1][y2] << "\n";
                        if (!d[x1][y2])
                        {
                            // cout << "passed: " << x1 + 1 << " " << y2 + 1 << "\n";
                            q.push(make_pair(x1, y2));
                            p[get_idx(x1, y2)] = get_idx(cell.first, cell.second);
                            d[x1][y2] = true;
                        }
                    }
                }
            }

            int y1 = cell.second + first_step[i];
            if (chk_bound(y1))
            {
                for (int j = 0; j < 2; j++)
                {
                    int x2 = cell.first + second_step[j];
                    if (chk_bound(x2))
                    {
                        if (!d[x2][y1])
                        {
                            q.push(make_pair(x2, y1));
                            p[get_idx(x2, y1)] = get_idx(cell.first, cell.second);
                            d[x2][y1] = true;
                        }
                    }
                }
            }

            
        }
        
    }

    return 0;
}