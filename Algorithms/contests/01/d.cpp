#include <bits/stdc++.h>

using namespace std;

#define ll long long
#define ul unsigned long
#define ull unsigned long long

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);

    ul q;    
    deque<ul> v, m;

    cin >> q;

    for (ul i = 0; i < q; i++){
        char c;
        cin >> c;
        if (c == '+'){
            ul a;
            cin >> a;
            v.push_back(a);
            if (m.empty()){
                m.push_front(a);
            }
            else{
                while(a < m.back()){
                    m.pop_back();
                    if (m.empty()){
                        break;
                    }
                }
                m.push_back(a);
            }
            cout << m.front() << "\n";
        }
        else if (c == '-'){
            if (v.front() == m.front()){
                m.pop_front();
            }
            v.pop_front();
            if (v.empty()){
                cout << "-1" << "\n";
            }
            else{
                cout << m.front() << "\n";
            }
        }
    }
    return 0;
}